"""This utility module contains functions to get the instances to explain for different datasets."""

from torch_geometric.data import DataLoader

from shapiq.explainer.graph.graph_datasets import CustomTUDataset


def _compute_baseline_value(x_graph):
    # This function computes the baseline value for the masked features, i.e. mean over nodes.
    return x_graph.x.mean(0)


def get_tu_instances(name):
    """Get the instances to explain for the given TU dataset."""
    dataset = CustomTUDataset(
        root="../shapiq/explainer/graph/graph_datasets",
        name=name,
        seed=1234,
        split_sizes=(0.8, 0.1, 0.1),
    )
    loader = DataLoader(dataset, shuffle=False)
    all_samples_to_explain = []
    for data in loader:
        for i in range(data.num_graphs):
            all_samples_to_explain.append(data[i])
    return all_samples_to_explain


def get_explanation_instances(dataset_name):
    """Get the instances to explain for the given dataset."""
    if dataset_name in [
        "AIDS",
        "DHFR",
        "COX2",
        "BZR",
        "PROTEINS",
        "ENZYMES",
        "MUTAG",
        "Mutagenicity",
    ]:
        all_samples_to_explain = get_tu_instances(dataset_name)
        return all_samples_to_explain
    raise ValueError("Dataset not supported.")

# Helper container with all the stored best configurations
_best_hyperparameters = {
        "GCN": {
                "AIDS": {},
                "DHFR": {},
                "COX2": {},
                "BZR": {},
                "PROTEINS": {},
                "ENZYMES": {},
                "MUTAG": {},
                "Mutagenicity": {"n_layers": {"3": {"hidden": 128}}},
        },
        }
