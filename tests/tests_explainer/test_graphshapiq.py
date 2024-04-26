from shapiq.explainer.graph.train_gnn import train_gnn
from graphxai_local.gnn_models.graph_classification import GCN_3layer, GCN_2layer, GCN_3layer_biased
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

import pytest

from shapiq.approximator import PermutationSamplingSII


def test_train_gnn():
    train_gnn()


@pytest.mark.parametrize(
    "CURRENT_MODEL_NAME",
    [("GCN3_bias")],
)
def test_graphshapiq_mutag(CURRENT_MODEL_NAME):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = MUTAG(root="", seed=1234, split_sizes=(0.8, 0.1, 0.1))
    dataset.graphs.data.to(device)
    num_nodes_features = dataset.graphs.num_node_features
    num_classes = dataset.graphs.num_classes
    test_loader = DataLoader(
        dataset[dataset.test_index], batch_size=len(dataset.test_index), shuffle=False
    )

    model_list = {}
    model_list["GCN3"] = GCN_3layer
    model_list["GCN3_bias"] = GCN_3layer_biased
    model_list["GCN2"] = GCN_2layer
    #model_list["GCN2_max"] = GCN_2layer_max
    #model_list["GCN3_mlp"] = GCN_3layer_mlp

    # Model and optimizer
    model = model_list[CURRENT_MODEL_NAME](num_nodes_features, 64, num_classes).to(device)
    model_path = (
        Path(__file__).resolve().parent.parent.parent
        / "graphxai_local"
        / "gnn_models"
        / "ckpt"
        / (CURRENT_MODEL_NAME + "_test.pth")
    )

    # Load best model
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    # Get the accurate samples on the test set (we explain only true positives for now)
    correct_samples = []
    for data in test_loader:  # Only one loop (#num_batches = 1)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct_samples = data[pred == data.y]

    # Get all samples with < 15 nodes from test set
    all_samples = []
    for data in test_loader:
        for i in range(data.num_graphs):
            if data[i].num_nodes <= 15:
                all_samples.append(data[i])
                pred = model(data[i].x, data[i].edge_index, data[i].batch)
                print(
                    f"{len(all_samples) - 1}) {data[i].num_nodes}, {data[i].y.item()}, {pred}, {torch.sigmoid(pred)}"
                )

    masking_mode = "feature-removal"
    EXPLANATION_ORDER = 2

    if CURRENT_MODEL_NAME in ["GCN3","GCN3_bias","GCN3_mlp"]:
        N_LAYERS = 3
    if CURRENT_MODEL_NAME in ["GCN2","GCN2_max"]:
        N_LAYERS = 2

    for data_id, x_graph in enumerate(all_samples):
        game = GraphGame(
            model,
            x_graph=x_graph,
            y_index=x_graph.y.item(),
            masking_mode=masking_mode,
            normalize=True,
        )
        # setup the explainer
        gSHAP = GraphSHAP(game.n_players, game, game.edge_index.numpy())
        exact_computer = ExactComputer(game.n_players, game)
        ground_truth = exact_computer.shapley_interaction(index="k-SII", order=EXPLANATION_ORDER)
        gt_moebius = exact_computer.moebius_transform()

        # Model output (logits) should be explained for all classes
        val_pred = game(game.grand_coalition)
        print("Predicted value to be explained: ", val_pred)

        mse_permutation = np.zeros(game.n_players + 1)
        mse = np.zeros(game.n_players + 1)
        mse_moebius = np.zeros(game.n_players + 1)
        model_calls_ratio = np.zeros(game.n_players + 1)
        neighbors = {}
        max_neighborhood_size = 0
        for node in gSHAP._grand_coalition_set:
            neighbors[node] = gSHAP._get_k_neighborhood(node, N_LAYERS)
            max_neighborhood_size = max(max_neighborhood_size, len(neighbors[node]))

        print("Number of nodes: ", game.n_players)
        print("Maxmimum Neighborhood size: ", max_neighborhood_size)

        for EFFICIENCY in [True, False]:
            for max_interaction_size in range(1, game.n_players + 1):
                # Compute the Moebius values for all subsets
                moebius, explanations = gSHAP.explain(
                    max_neighborhood_size=N_LAYERS,
                    max_interaction_size=max_interaction_size,
                    order=EXPLANATION_ORDER,
                    efficiency_routine=EFFICIENCY,
                )
                model_calls_ratio[max_interaction_size] = (
                    gSHAP.last_n_model_calls / 2**game.n_players
                )
                mse[max_interaction_size] = np.sum((explanations - ground_truth).values ** 2)
                mse_moebius[max_interaction_size] = np.sum((gt_moebius - moebius).values ** 2)
                print("Max Moebius Size: ", max_interaction_size)
                print(" error explaination: ", mse[max_interaction_size])
                print(
                    " error Moebius :",
                    mse_moebius[max_interaction_size],
                )

                permutation_sampler = PermutationSamplingSII(
                    n=game.n_players, max_order=EXPLANATION_ORDER, index="k-SII"
                )
                sampling_approximation = permutation_sampler.approximate(
                    game=game, budget=gSHAP.last_n_model_calls
                )
                mse_permutation[max_interaction_size] = np.sum(
                    (sampling_approximation - ground_truth).values ** 2
                )

            save_path = Path(__file__).resolve().parent.parent.parent / "results" / CURRENT_MODEL_NAME

            id = "_".join([CURRENT_MODEL_NAME,str(data_id), str(game.n_players), str(EFFICIENCY)])

            results = pd.DataFrame(
                index=range(game.n_players + 1),
                data=np.vstack((model_calls_ratio, mse, mse_moebius, mse_permutation)).T,
                columns=("model_calls_ratio", "mse", "mse_moebius", "mse_permutation"),
            )
            results.index.name = "max_moebius_size"
            results["max_neighborhood_size"] = max_neighborhood_size
            results.to_csv(save_path / ("results_" + id))


@pytest.mark.parametrize(
    "CURRENT_MODEL_NAME",
    [("GCN3_bias")],
)
def test_plot_results(CURRENT_MODEL_NAME):
    save_directory = Path(__file__).resolve().parent.parent.parent / "results" / CURRENT_MODEL_NAME
    save_path_plots = save_directory / "plots"
    for file_path in save_directory.glob(("results_"+CURRENT_MODEL_NAME+"*")):
        results = pd.read_csv(file_path)
        plot_range = range(results.index.min(), results.index.max() + 1)

        id = file_path.name[8:]
        max_neighborhood_size = results.loc[
            results.max_neighborhood_size.max()
        ].model_calls_ratio


        def plot_figure(
            x, y1, y1_label, label, title, y2=None, y2_label=None
        ):
            plt.figure()
            plt.plot(
                x,
                y1,
                marker="o",
                label=y1_label,
            )
            if y2 is not None:
                plt.plot(
                    x,
                    y2,
                    marker="o",
                    label=y2_label,
                )
            plt.legend()
            plt.title(title)
            plt.savefig(save_path_plots / label)
            plt.show()

        # plot explanation SSE
        plot_figure(
            x=results.model_calls_ratio[plot_range] * 100,
            y1=results.mse[plot_range],
            y1_label="GraphSHAP",
            y2=results.mse_permutation[plot_range],
            y2_label="Permutation",
            label="explanation_" + id,
            title="Explanation SSE " + id,
        )
        # plot explanation log SSE
        plot_figure(
            x=results.model_calls_ratio[plot_range] * 100,
            y1=np.log(results.mse[plot_range]),
            y1_label="GraphSHAP",
            y2=np.log(results.mse_permutation[plot_range]),
            y2_label="Permutation",
            label="explanation_log_" + id,
            title="Explanation log SSE " + id,
        )
        # plot moebius log SSE
        plot_figure(
            x=results.model_calls_ratio[plot_range] * 100,
            y1=np.log(results.mse_moebius[plot_range]),
            y1_label="Moebius",
            label="moebius_log_" + id,
            title="Moebius log SSE " + id,
        )