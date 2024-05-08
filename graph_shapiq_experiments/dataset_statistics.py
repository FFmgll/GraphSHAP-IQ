from shapiq.explainer.graph import get_explanation_instances
from torch_geometric.utils import to_networkx
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    DATASET_NAMES = [
        "AIDS",
        "DHFR",
        "COX2",
        "BZR",
        "PROTEINS",
        "ENZYMES",
        "MUTAG",
        "Mutagenicity",
    ]  # ["AIDS","DHFR","COX2","BZR","PROTEINS", "ENZYMES", "MUTAG", "Mutagenicity"]


    for dataset_name in DATASET_NAMES:
        all_samples_to_explain = get_explanation_instances(dataset_name)
        num_nodes = {}
        num_edges = {}
        avg_node_degree = {}
        min_node_degree = {}
        max_node_degree = {}
        graph_density = {}

        for data_id, explain_instance in enumerate(all_samples_to_explain):
            num_nodes[data_id] = explain_instance.num_nodes
            g = to_networkx(explain_instance, to_undirected=False)  # converts into graph
            avg_node_degree[data_id] = np.mean(g.degree())
            max_node_degree[data_id] = np.max(g.degree())
            min_node_degree[data_id] = np.min(g.degree())
            num_edges[data_id] = explain_instance.num_edges
            graph_density[data_id] = 2*explain_instance.num_edges/(explain_instance.num_nodes*(explain_instance.num_nodes-1))

        df = pd.DataFrame(index=num_nodes.keys(), data=num_nodes.values())
        df["avg_node_degree"] = avg_node_degree
        df["min_node_degree"] = min_node_degree
        df["max_node_degree"] = max_node_degree
        df["num_edges"] = num_edges
        df["graph_density"] = graph_density
        save_path = os.path.join("../results/complexity_analysis/dataset_statistics", dataset_name + ".csv")
        df.to_csv(save_path)