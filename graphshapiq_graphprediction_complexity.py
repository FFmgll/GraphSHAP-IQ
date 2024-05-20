import pandas as pd

from shapiq.explainer.graph import GraphSHAPIQ
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.explainer.graph import get_explanation_instances

import torch
import numpy as np
import os
import tqdm


class dummyModel(torch.nn.Module):
    def __init__(self, num_classes=20):
        super(dummyModel, self).__init__()
        self.num_classes = 20
    def forward(self, x, edge_index, batch):
        # Output dummy predictions for number of classes
        return torch.zeros((1,self.num_classes), dtype=float)



def evaluate_complexity(
    dataset_name, n_layers, all_samples_to_explain, masking_mode="feature-removal"
):
    gshap_budget = {}
    gshap_budget_est = {}
    players = {}
    max_neighborhood_size = {}

    for data_id, x_graph in tqdm.tqdm(enumerate(all_samples_to_explain),total=len(all_samples_to_explain),desc=dataset_name+" ("+str(n_layers)+" Layers)"):
        dummy_model = dummyModel()
        game = GraphGame(
            dummy_model,
            x_graph=x_graph,
            class_id=x_graph.y.item(),
            max_neighborhood_size=n_layers,
            masking_mode=masking_mode,
            normalize=False,
            baseline=0,
        )
        # setup the explainer
        gSHAP = GraphSHAPIQ(game)

        gshap_budget[data_id] = gSHAP.total_budget
        players[data_id] = gSHAP.n_players
        max_neighborhood_size[data_id] = gSHAP.max_neighborhood_size
        gshap_budget_est[data_id] = gSHAP.budget_estimated

    results = pd.DataFrame(
        index=gshap_budget.keys(), data=gshap_budget.values(), columns=["budget"]
    )
    results["budget_upper_bound"] = gshap_budget_est
    results["max_neighborhood_size"] = max_neighborhood_size
    results["n_players"] = players

    save_name = "_".join(["complexity", dataset_name + "_" + str(n_layers)])
    save_path = os.path.join("results/complexity_analysis", save_name + ".csv")
    results.to_csv(save_path)


if __name__ == "__main__":
    DATASET_NAMES = [
         "COX2",
         "BZR",
         "PROTEINS",
         "ENZYMES",
         "Mutagenicity",
        "FluorideCarbonyl",
        "Benzene",
        "AlkaneCarbonyl",
    ]  # ["AIDS","DHFR","COX2","BZR","PROTEINS", "ENZYMES", "MUTAG", "Mutagenicity"]
    N_LAYERS = [1, 2, 3, 4]

    for dataset_name in DATASET_NAMES:
        all_samples_to_explain = get_explanation_instances(dataset_name)
        for n_layers in N_LAYERS:
            evaluate_complexity(dataset_name, n_layers, all_samples_to_explain)
