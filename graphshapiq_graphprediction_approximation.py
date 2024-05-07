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
import matplotlib.pyplot as plt

import os

from shapiq.approximator import (
    PermutationSamplingSII,
    KernelSHAPIQ,
    InconsistentKernelSHAPIQ,
    SVARMIQ,
    SHAPIQ,
)


def get_MUTAG_instances():
    # Load dataset
    dataset = MUTAG(
        root="shapiq/explainer/graph/graph_datasets", seed=1234, split_sizes=(0.8, 0.1, 0.1)
    )
    dataset.graphs.data.to(device)
    loader = DataLoader(dataset, shuffle=False)

    # Get the accurate samples on the test set (we explain only true positives for now)
    correct_samples = []
    # for data in test_loader:  # Only one loop (#num_batches = 1)
    # out = model(data.x, data.edge_index, data.batch)
    # pred = out.argmax(dim=1)

    # Get all samples with < 15 nodes from test set
    all_samples_to_explain = []
    for data in loader:
        for i in range(data.num_graphs):
            # if data[i].num_nodes <= 12:
            all_samples_to_explain.append(data[i])

    return all_samples_to_explain


def get_TU_instances(name):
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
    # if dataset_name == "MUTAG":
    #    all_samples_to_explain = get_MUTAG_instances()
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


def save_results(identifier, model_id, data_id, game, sse, max_neighborhood_size, gshap_budget):
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
    for i, approx_id in enumerate(sse.keys()):
        tmp = pd.DataFrame(
            index=sse[approx_id].keys(), data=sse[approx_id].values(), columns=[approx_id]
        )
        if i == 0:
            final_results = tmp
        else:
            final_results = final_results.join(tmp)

    final_results["budget_with_efficiency"] = gshap_budget[True]
    final_results["budget_no_efficiency"] = gshap_budget[False]

    save_name = "_".join(
        [
            identifier,
            model_id,
            str(data_id),
            str(game.n_players),
            str(max_neighborhood_size),
        ]
    )
    if identifier == "withGT":
        save_path = os.path.join("results/approximation_with_gt", save_name + ".csv")
    if identifier == "noGT":
        save_path = os.path.join("results/approximation_without_gt", save_name + ".csv")
    final_results.to_csv(save_path)


def explain_instances_with_gt(
    model_id, model, all_samples_to_explain, explanation_order=2, masking_mode="feature-removal"
):
    BASELINE_APPROXIMATORS = {
        "Permutation": PermutationSamplingSII,
        "KernelSHAPIQ": KernelSHAPIQ,
        "incKernelSHAPIQ": InconsistentKernelSHAPIQ,
        "SHAPIQ": SHAPIQ,
        "SVARMIQ": SVARMIQ,
    }
    EFFICIENCY_MODES = [True, False]
    stop_processing = False
    counter = 0
    print("Running with GT ...", model_id)

    for data_id, x_graph in enumerate(all_samples_to_explain):
        if not stop_processing:
            baseline = x_graph.x.mean(0)
            if x_graph.num_nodes <= 12:
                # print(x_graph.num_nodes)
                game = GraphGame(
                    model,
                    x_graph=x_graph,
                    class_id=x_graph.y.item(),
                    max_neighborhood_size=model.n_layers,
                    masking_mode=masking_mode,
                    normalize=True,
                    baseline=baseline,
                )
                # setup the explainer
                gSHAP = GraphSHAPIQ(game)

                exact_computer = ExactComputer(game.n_players, game)
                gt_interaction = exact_computer.shapley_interaction(
                    index="k-SII", order=explanation_order
                )
                gt_moebius = exact_computer.moebius_transform()

                sse = {}
                gshap_budget = {}
                for efficiency_mode in EFFICIENCY_MODES:
                    approx_id_interaction = "GraphSHAPIQ_" + str(efficiency_mode) + "_interaction"
                    approx_id_moebius = "GraphSHAPIQ_" + str(efficiency_mode) + "_moebius"

                    gshap_budget[efficiency_mode] = {}

                    sse[approx_id_interaction] = {}
                    sse[approx_id_moebius] = {}
                    for max_interaction_size in range(1, gSHAP.max_size_neighbors + 1):
                        # Compute the Moebius values for all subsets
                        gshap_moebius, gshap_interactions = gSHAP.explain(
                            max_interaction_size=max_interaction_size,
                            order=explanation_order,
                            efficiency_routine=efficiency_mode,
                        )
                        gshap_budget[efficiency_mode][
                            max_interaction_size
                        ] = gSHAP.last_n_model_calls
                        sse[approx_id_interaction][max_interaction_size] = np.sum(
                            (gshap_interactions - gt_interaction).values ** 2
                        )
                        sse[approx_id_moebius][max_interaction_size] = np.sum(
                            (gt_moebius - gshap_moebius).values ** 2
                        )

                    for approx_id, baseline in BASELINE_APPROXIMATORS.items():
                        baseline_approx = baseline(
                            n=game.n_players, max_order=explanation_order, index="k-SII"
                        )
                        sse[approx_id] = {}
                        for max_interaction_size, budget in gshap_budget[True].items():
                            baseline_interactions = baseline_approx.approximate(
                                game=game, budget=budget
                            )
                            sse[approx_id][max_interaction_size] = np.sum(
                                (gt_interaction - baseline_interactions).values ** 2
                            )

                save_results(
                    "withGT", model_id, data_id, game, sse, gSHAP.max_size_neighbors, gshap_budget
                )
                counter += 1
        if counter >= 5:
            stop_processing = True


def explain_instances(
    model_id, model, all_samples_to_explain, explanation_order=2, masking_mode="feature-removal"
):
    BASELINE_APPROXIMATORS = {
        "Permutation": PermutationSamplingSII,
        "KernelSHAPIQ": KernelSHAPIQ,
        "incKernelSHAPIQ": InconsistentKernelSHAPIQ,
        "SHAPIQ": SHAPIQ,
        "SVARMIQ": SVARMIQ,
    }
    EFFICIENCY_MODES = [True, False]
    stop_processing = False
    counter = 0
    print("Running without GT...", model_id)
    for data_id, x_graph in enumerate(all_samples_to_explain):
        if not stop_processing and x_graph.num_nodes >= 12:
            baseline = x_graph.x.mean(0)
            game = GraphGame(
                model,
                x_graph=x_graph,
                class_id=x_graph.y.item(),
                max_neighborhood_size=model.n_layers,
                masking_mode=masking_mode,
                normalize=True,
                baseline=baseline,
            )
            # setup the explainer
            gSHAP = GraphSHAPIQ(game)
            if gSHAP.total_budget <= 2**12:
                # print(x_graph.num_nodes, gSHAP.max_size_neighbors)

                sse = {}
                gshap_budget = {}
                gshap_moebius = {}
                gshap_interactions = {}

                for efficiency_mode in EFFICIENCY_MODES:
                    approx_id_interaction = "GraphSHAPIQ_" + str(efficiency_mode) + "_interaction"
                    approx_id_moebius = "GraphSHAPIQ_" + str(efficiency_mode) + "_moebius"

                    gshap_budget[efficiency_mode] = {}

                    sse[approx_id_interaction] = {}
                    sse[approx_id_moebius] = {}
                    for max_interaction_size in range(1, gSHAP.max_size_neighbors + 1):
                        # print("GSHAP running...", max_interaction_size)
                        # Compute the Moebius values for all subsets
                        (
                            gshap_moebius[max_interaction_size],
                            gshap_interactions[max_interaction_size],
                        ) = gSHAP.explain(
                            max_interaction_size=max_interaction_size,
                            order=explanation_order,
                            efficiency_routine=efficiency_mode,
                        )
                        gshap_budget[efficiency_mode][
                            max_interaction_size
                        ] = gSHAP.last_n_model_calls

                    gt_interaction = gshap_interactions[gSHAP.max_size_neighbors]
                    gt_moebius = gshap_moebius[gSHAP.max_size_neighbors]

                    for max_interaction_size in range(1, gSHAP.max_size_neighbors + 1):
                        # Compute SSE
                        sse[approx_id_interaction][max_interaction_size] = np.sum(
                            (gshap_interactions[max_interaction_size] - gt_interaction).values ** 2
                        )
                        sse[approx_id_moebius][max_interaction_size] = np.sum(
                            (gt_moebius - gshap_moebius[max_interaction_size]).values ** 2
                        )

                    for approx_id, baseline in BASELINE_APPROXIMATORS.items():
                        baseline_approx = baseline(
                            n=game.n_players, max_order=explanation_order, index="k-SII"
                        )
                        sse[approx_id] = {}
                        for max_interaction_size, budget in gshap_budget[True].items():
                            # print("Baseline " + approx_id + " ...", max_interaction_size)
                            baseline_interactions = baseline_approx.approximate(
                                game=game, budget=budget
                            )
                            sse[approx_id][max_interaction_size] = np.sum(
                                (gt_interaction - baseline_interactions).values ** 2
                            )

                save_results(
                    "noGT", model_id, data_id, game, sse, gSHAP.max_size_neighbors, gshap_budget
                )
                counter += 1
        if counter >= 5:
            stop_processing = True


if __name__ == "__main__":
    DATASET_NAMES = ["PROTEINS", "ENZYMES", "MUTAG", "AIDS", "DHFR", "COX2", "BZR", "Mutagenicity"]
    MODEL_TYPES = ["GCN"]
    N_LAYERS = [1, 2, 3, 4]
    NODE_BIASES = [True]  # [False,True]
    GRAPH_BIASES = [True]  # [False,True]
    EXPLAIN_WITH_GT = True
    EXPLAIN_WITHOUT_GT = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in DATASET_NAMES:
        all_samples_to_explain = get_explanation_instances(dataset_name)
        for model_type in MODEL_TYPES:
            for n_layers in N_LAYERS:
                for node_bias in NODE_BIASES:
                    for graph_bias in GRAPH_BIASES:
                        model, model_id = train_gnn(
                            dataset_name=dataset_name,
                            model_type=model_type,
                            n_layers=n_layers,
                            node_bias=node_bias,
                            graph_bias=graph_bias,
                        )
                        model.eval()
                        if EXPLAIN_WITH_GT:
                            explain_instances_with_gt(model_id, model, all_samples_to_explain)
                        if EXPLAIN_WITHOUT_GT:
                            explain_instances(model_id, model, all_samples_to_explain)
