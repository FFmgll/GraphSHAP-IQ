from shapiq.explainer.graph.train_gnn import train_gnn
from graphxai_local.datasets.real_world.MUTAG import (
    MUTAG,
)  # renamed to avoid conflict with potential installs
from pathlib import Path
from torch_geometric.loader import DataLoader

# import modules
import torch
import pandas as pd

from shapiq import ExactComputer
from shapiq.explainer.graph import GraphSHAP
from shapiq.games.benchmark.local_xai import GraphGame

import numpy as np
import matplotlib.pyplot as plt

import os

from shapiq.approximator import PermutationSamplingSII, KernelSHAPIQ, InconsistentKernelSHAPIQ, SVARMIQ, SHAPIQ

def get_MUTAG_instances():
    # Load dataset
    dataset = MUTAG(root="", seed=1234, split_sizes=(0.8, 0.1, 0.1))
    dataset.graphs.data.to(device)
    test_loader = DataLoader(
        dataset[dataset.test_index], batch_size=len(dataset.test_index), shuffle=False
    )

    # Get the accurate samples on the test set (we explain only true positives for now)
    correct_samples = []
    #for data in test_loader:  # Only one loop (#num_batches = 1)
        #out = model(data.x, data.edge_index, data.batch)
        #pred = out.argmax(dim=1)

    # Get all samples with < 15 nodes from test set
    all_samples_to_explain = []
    for data in test_loader:
        for i in range(data.num_graphs):
            if data[i].num_nodes <= 14:
                all_samples_to_explain.append(data[i])

    return all_samples_to_explain


def get_explanation_instances(dataset_name="MUTAG"):
    if dataset_name == "MUTAG":
        all_samples_to_explain = get_MUTAG_instances()
    return all_samples_to_explain


def save_results(model_id,data_id,game,sse,max_neighborhood_size,gshap_budget):
    """
    Naming convention for results file are the following attributes separated by underscore:
        - Type of model, e.g. GCN, GIN
        - Dataset name, e.g. MUTAG
        - Number of Graph Convolutions, e.g. 2 graph conv layers
        - Graph bias, e.g. True, if the linear layer after global pooling has a bias
        - Node bias, e.g. True, if the convolution layers have a bias
        - Data ID: a technical identifier of the explained instance
        - Number of players, i.e. number of nodes in the graph
        - Largest neighborhood size as integer
    """
    for i,approx_id in enumerate(sse.keys()):
        tmp = pd.DataFrame(index=sse[approx_id].keys(),data=sse[approx_id].values(),columns=[approx_id])
        if i==0:
            final_results = tmp
        else:
            final_results = final_results.join(tmp)

    final_results["budget_with_efficiency"] = gshap_budget[True]
    final_results["budget_no_efficiency"] = gshap_budget[False]

    save_name = "_".join([model_id,str(data_id),str(game.n_players),str(max_neighborhood_size)])
    save_path =os.path.join("results", save_name+".csv")
    final_results.to_csv(save_path)


def explain_instances(model_id, model, all_samples_to_explain,explanation_order=2,masking_mode="feature-removal"):
    BASELINE_APPROXIMATORS = {"Permutation":PermutationSamplingSII, "KernelSHAPIQ": KernelSHAPIQ, "incKernelSHAPIQ": InconsistentKernelSHAPIQ, "SHAPIQ": SHAPIQ, "SVARMIQ": SVARMIQ}
    EFFICIENCY_MODES = [True,False]

    for data_id, x_graph in enumerate(all_samples_to_explain):
        game = GraphGame(
            model,
            x_graph=x_graph,
            y_index=x_graph.y.item(),
            masking_mode=masking_mode,
            normalize=True,
        )
        # setup the explainer
        gSHAP = GraphSHAP(game.n_players, game, game.edge_index.numpy(),n_layers=model.n_layers)
        exact_computer = ExactComputer(game.n_players, game)
        gt_interaction = exact_computer.shapley_interaction(index="k-SII", order=explanation_order)
        gt_moebius = exact_computer.moebius_transform()

        sse = {}
        neighbors = {}
        max_neighborhood_size = 0
        for node in gSHAP._grand_coalition_set:
            neighbors[node] = gSHAP._get_k_neighborhood(node, model.n_layers)
            max_neighborhood_size = max(max_neighborhood_size, len(neighbors[node]))

        gshap_budget = {}
        for efficiency_mode in EFFICIENCY_MODES:
            approx_id_interaction = "GraphSHAPIQ_" + str(efficiency_mode) + "_interaction"
            approx_id_moebius = "GraphSHAPIQ_" + str(efficiency_mode) + "_moebius"

            gshap_budget[efficiency_mode] = {}

            sse[approx_id_interaction] = {}
            sse[approx_id_moebius] = {}
            for max_interaction_size in range(1,max_neighborhood_size + 1):
                    # Compute the Moebius values for all subsets
                    gshap_moebius, gshap_interactions = gSHAP.explain(
                        max_interaction_size=max_interaction_size,
                        order=explanation_order,
                        efficiency_routine=efficiency_mode,
                    )
                    gshap_budget[efficiency_mode][max_interaction_size] = gSHAP.last_n_model_calls
                    sse[approx_id_interaction][max_interaction_size] = np.sum((gshap_interactions - gt_interaction).values ** 2)
                    sse[approx_id_moebius][max_interaction_size] = np.sum((gt_moebius - gshap_moebius).values ** 2)

            for approx_id, baseline in BASELINE_APPROXIMATORS.items():
                baseline_approx = baseline(n=game.n_players, max_order=explanation_order, index="k-SII")
                sse[approx_id] = {}
                for max_interaction_size, budget in gshap_budget[True].items():
                    baseline_interactions = baseline_approx.approximate(game=game, budget=budget)
                    sse[approx_id][max_interaction_size] = np.sum((gt_interaction-baseline_interactions).values**2)

        save_results(model_id,data_id,game,sse,max_neighborhood_size,gshap_budget)


if __name__ == "__main__":
    DATASET_NAMES = ["MUTAG"]
    MODEL_TYPES = ["GCN","GIN"]
    N_LAYERS = [2,3]
    NODE_BIASES = [False,True]
    GRAPH_BIASES = [False,True]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_samples_to_explain = get_explanation_instances("MUTAG")


    for dataset_name in DATASET_NAMES:
        for model_type in MODEL_TYPES:
            for n_layers in N_LAYERS:
                for node_bias in NODE_BIASES:
                    for graph_bias in GRAPH_BIASES:
                        model, model_id = train_gnn(dataset_name=dataset_name, model_type=model_type, n_layers=n_layers, node_bias=node_bias, graph_bias=graph_bias)
                        explain_instances(model_id,model,all_samples_to_explain)


