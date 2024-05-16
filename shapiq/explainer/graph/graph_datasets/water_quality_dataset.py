from torch_geometric.data import Dataset, Data, DataLoader
from pathlib import Path
import numpy as np
import torch
import os

this_file_path = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(this_file_path, "data", "WaterQuality")


class WaterQuality(Dataset):
    """
    A simple water quality dataset. The graph is a water distribution system (WDS)
    where nodes correspond to consumers and water reservoirs, and edges correspond
    to the pipes of the system. Chlorine is injected into the reservoir and distributes
    according to water flow (advection) and diffusion. Advection is the dominating
    factor. The task is to predict the total fraction of chlorinated nodes at
    some point in the future.
    """

    def __init__(self, coverage_forecast_steps=4, subset="train"):
        """
        Load dataset into memory.
        coverage_forecast_steps indicates how many steps into the future chlorine
        concentration should be prediced.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.coverage_forecast_steps = coverage_forecast_steps
        self.subset = subset
        self.subset_ids = {}
        self.subset_ids["train"] = np.random.RandomState(42).choice(1000, size=800, replace=False)
        self.subset_ids["test"] = set(range(10)) - set(self.subset_ids["train"])
        self.load()

    def filter_subset(self):
        return filter(
            lambda p: int(p.stem) in self.subset_ids[self.subset],
            Path(DATA_DIR).rglob("[0-9]*.npz"),
        )

    def load(self):
        self.quality_data = []
        self.flow_data = []
        graph = np.load(os.path.join(DATA_DIR, "graph.npz"), allow_pickle=True)
        self.edge_index = graph["edge_index"]
        self.edge_index = np.concatenate([self.edge_index, self.edge_index[::-1]], axis=-1)
        self.edge_index = torch.tensor(self.edge_index).to(device=self.device)
        self.node_pos = graph["pos"].item()
        self.n_scenarios = 0
        self._len = 0
        self.quality_max = -1e10
        self.flow_max = -1e10
        self.steps_per_sample = 36

        for file in self.filter_subset():
            if str(file).endswith("graph.npz"):
                continue
            raw = np.load(file)
            x = raw["node_quality"]
            flows = np.concatenate([raw["flow_data"], -raw["flow_data"]], axis=-1)
            self.quality_data.append(torch.tensor(x).float().to(device=self.device))
            self.flow_data.append(torch.tensor(flows).float().to(device=self.device))
            self._len += len(x) - self.coverage_forecast_steps - 1
            self.n_scenarios += 1
            self.quality_max = max(self.quality_max, np.max(x))
            self.flow_max = max(self.quality_max, np.max(flows))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        sid = idx // (self.steps_per_sample - self.coverage_forecast_steps)
        tid = idx % (self.steps_per_sample - self.coverage_forecast_steps)
        y = self.quality_data[sid][tid + self.coverage_forecast_steps]
        y = y.mean()[None, None]
        return Data(
            x=self.quality_data[sid][tid][:, None] / self.quality_max,
            edge_features=self.flow_data[sid][tid][:, None] / self.flow_max,
            edge_index=self.edge_index,
            y=torch.tensor([[0]]).to(device=self.device),
            label=y,
            time=tid,
            num_nodes=32,
        ).to("cpu")
