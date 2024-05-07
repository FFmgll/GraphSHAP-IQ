from torch_geometric.data import DataLoader
from shapiq.explainer.graph.graph_datasets import CustomTUDataset


def _compute_baseline_value(x_graph):
    # This function computes the baseline value for the masked features, i.e. mean over nodes.
    return x_graph.x.mean(0)

def get_TU_instances(name):
    # These are the TUDataset instances, which we will explain
    dataset = CustomTUDataset(
        root="shapiq/explainer/graph/graph_datasets",
        name=name,
        seed=1234,
        split_sizes=(0.8, 0.1, 0.1),
    )
    loader = DataLoader(dataset, shuffle=False)
    # Get all samples with < 15 nodes from test set
    all_samples_to_explain = []
    for data in loader:
        for i in range(data.num_graphs):
            # if data[i].num_nodes <= 60:
            all_samples_to_explain.append(data[i])
    return all_samples_to_explain



def get_explanation_instances(dataset_name):
    # These are the instances we will explain for different datasets
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
        all_samples_to_explain = get_TU_instances(dataset_name)
    return all_samples_to_explain