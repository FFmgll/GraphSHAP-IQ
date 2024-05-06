from shapiq.explainer.graph.train_gnn import train_gnn
from graphxai_local.datasets.real_world.MUTAG import (
    MUTAG,
)  # renamed to avoid conflict with potential installs
from pathlib import Path
from torch_geometric.loader import DataLoader

from datetime import datetime

# import modules
import torch
import pandas as pd

from shapiq import ExactComputer
from shapiq.explainer.graph import GraphSHAPIQ
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.explainer.graph.graph_datasets import CustomTUDataset
import numpy as np
import os


def get_MUTAG_instances():
    # Load dataset
    dataset = MUTAG(root="shapiq/explainer/graph/graph_datasets", seed=1234, split_sizes=(0.8, 0.1, 0.1))
    dataset.graphs.data.to(device)
    loader = DataLoader(
        dataset, shuffle=False
    )
    # Get all samples with < 15 nodes from test set
    all_samples_to_explain = []
    for data in loader:
        for i in range(data.num_graphs):
            all_samples_to_explain.append(data[i])
    return all_samples_to_explain

def get_TU_instances(name):
    dataset = CustomTUDataset(root="shapiq/explainer/graph/graph_datasets", name=name, seed=1234, split_sizes=(0.8, 0.1, 0.1))
    loader = DataLoader(dataset, shuffle=False)
    # Get all samples with < 15 nodes from test set
    all_samples_to_explain = []
    for data in loader:
        for i in range(data.num_graphs):
            all_samples_to_explain.append(data[i])
    return all_samples_to_explain



def get_explanation_instances(dataset_name):
    #if dataset_name == "MUTAG":
    #    all_samples_to_explain = get_MUTAG_instances()
    if dataset_name in ["AIDS","DHFR","COX2","BZR","PROTEINS", "ENZYMES", "MUTAG", "Mutagenicity"]:
        all_samples_to_explain = get_TU_instances(dataset_name)
    return all_samples_to_explain


def evaluate_complexity(model_id, model, all_samples_to_explain,masking_mode="feature-removal"):
    complexity_results = np.zeros((len(all_samples_to_explain),5))
    counter = 0
    for data_id, x_graph in enumerate(all_samples_to_explain):
        game = GraphGame(
            model,
            x_graph=x_graph,
            class_id=x_graph.y.item(),
            max_neighborhood_size=model.n_layers,
            masking_mode=masking_mode,
            normalize=True,
        )
        # setup the explainer
        gSHAP = GraphSHAPIQ(game)

        for node in gSHAP.neighbors:
            complexity_results[data_id,0] += 2**len(gSHAP.neighbors[node])
        complexity_results[data_id,2] = game.n_players
        complexity_results[data_id,3] = gSHAP.max_size_neighbors
        complexity_results[data_id,1] = gSHAP.total_budget
        complexity_results[data_id,4] = gSHAP.budget_estimated

        results = pd.DataFrame(complexity_results[:counter,:], columns=["bound_gSHAP","exact_gSHAP","n_players","max_size_neighbors","budget_estimated"])
        results["log10_budget_ratio_perc"] = np.log10(results["exact_gSHAP"])-results["n_players"]*np.log10(2) + 2

        save_name = "_".join(["complexity",model_id])
        save_path = os.path.join("results", save_name + ".csv")
        results.to_csv(save_path)
        counter += 1


if __name__ == "__main__":
    DATASET_NAMES = ["AIDS","DHFR","COX2","BZR","PROTEINS", "ENZYMES", "MUTAG", "Mutagenicity"] # ["AIDS","DHFR","COX2","BZR","PROTEINS", "ENZYMES", "MUTAG", "Mutagenicity"]
    MODEL_TYPES = ["GCN"]
    N_LAYERS = [1,2,3,4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NODE_BIAS = True
    GRAPH_BIAS = True

    for dataset_name in DATASET_NAMES:
        all_samples_to_explain = get_explanation_instances(dataset_name)
        for model_type in MODEL_TYPES:
            for n_layers in N_LAYERS:
                model, model_id = train_gnn(dataset_name=dataset_name, model_type=model_type, n_layers=n_layers, node_bias=NODE_BIAS, graph_bias=GRAPH_BIAS)
                model.eval()
                evaluate_complexity(model_id, model, all_samples_to_explain)

