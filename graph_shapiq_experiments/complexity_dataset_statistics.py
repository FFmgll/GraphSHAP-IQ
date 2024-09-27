from shapiq.explainer.graph import get_explanation_instances
from torch_geometric.utils import to_networkx
import numpy as np
import pandas as pd
import os
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import tqdm
from shapiq.explainer.graph.utils import get_water_quality_graph


if __name__ == "__main__":
    DATASET_NAMES = [
        #"COX2",
        #"BZR",
        #"PROTEINS",
        #"ENZYMES",
        #"Mutagenicity",
        #"FluorideCarbonyl",
        #"Benzene",
        "AlkaneCarbonyl",
        #"WaterQuality"
    ]  # ["AIDS","DHFR","COX2","BZR","PROTEINS", "ENZYMES", "MUTAG", "Mutagenicity", "FluorideCarbonyl", "Benzene", "AlkaneCarbonyl",]

    for dataset_name in DATASET_NAMES:
        if dataset_name == "WaterQuality":
            # Get single graph for water quality
            all_samples_to_explain = get_water_quality_graph()
        else:
            all_samples_to_explain = get_explanation_instances(dataset_name)
        num_nodes = {}
        num_edges = {}
        avg_node_degree = {}
        min_node_degree = {}
        max_node_degree = {}
        avg_curvature = {}
        min_curvature = {}
        max_curvature = {}
        graph_density = {}
        for data_id, explain_instance in tqdm.tqdm(
            enumerate(all_samples_to_explain), total=len(all_samples_to_explain), desc=dataset_name
        ):
            num_nodes[data_id] = explain_instance.num_nodes
            g = to_networkx(explain_instance, to_undirected=True)  # converts into graph
            curvature = OllivierRicci(g)
            curvature_graph = curvature.compute_ricci_curvature()
            g_curvature = np.zeros(len(curvature_graph.nodes))
            for i, curvature_data in curvature_graph.nodes(data=True):
                if "ricciCurvature" in curvature_data:
                    g_curvature[i] = curvature_data["ricciCurvature"]
                else:
                    g_curvature[i] = 0
            avg_curvature[data_id] = np.mean(g_curvature)
            min_curvature[data_id] = np.min(g_curvature)
            max_curvature[data_id] = np.max(g_curvature)

            node_degrees = dict(g.degree()).values()
            avg_node_degree[data_id] = np.mean(list(node_degrees))
            max_node_degree[data_id] = np.max(node_degrees)
            min_node_degree[data_id] = np.min(node_degrees)
            num_edges[data_id] = explain_instance.num_edges
            graph_density[data_id] = (
                2
                * explain_instance.num_edges
                / (explain_instance.num_nodes * (explain_instance.num_nodes - 1))
            )

        df = pd.DataFrame(index=num_nodes.keys(), data=num_nodes.values())
        df["avg_node_degree"] = avg_node_degree
        df["min_node_degree"] = min_node_degree
        df["max_node_degree"] = max_node_degree
        df["avg_curvature"] = avg_curvature
        df["min_curvature"] = min_curvature
        df["max_curvature"] = max_curvature
        df["num_edges"] = num_edges
        df["graph_density"] = graph_density
        save_path = os.path.join(
            "../results/complexity_analysis/dataset_statistics", dataset_name + ".csv"
        )
        df.to_csv(save_path)
