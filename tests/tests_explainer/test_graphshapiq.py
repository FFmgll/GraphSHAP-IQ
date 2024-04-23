from shapiq.explainer.graph.train_gnn import train_gnn
from graphxai_local.gnn_models.graph_classification import GCN_3layer
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
    [("GCN3")],
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
    # Model and optimizer
    model = GCN_3layer(num_nodes_features, 64, num_classes).to(device)
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
    EFFICIENCY = False

    for i, x_graph in enumerate(all_samples):
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
            neighbors[node] = gSHAP._get_k_neighborhood(node, 3)
            max_neighborhood_size = max(max_neighborhood_size, len(neighbors[node]))

        print("Number of nodes: ", game.n_players)
        print("Maxmimum Neighborhood size: ", max_neighborhood_size)

        for max_interaction_size in range(1, max_neighborhood_size + 1):
            # Compute the Moebius values for all subsets
            moebius, explanations = gSHAP.explain(
                max_neighborhood_size=3,
                max_interaction_size=max_interaction_size,
                order=EXPLANATION_ORDER,
                efficiency_routine=EFFICIENCY,
            )
            model_calls_ratio[max_interaction_size] = gSHAP.last_n_model_calls / 2**game.n_players
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

        plot_range = range(1,max_neighborhood_size + 1)

        save_path = Path(__file__).resolve().parent.parent.parent / "results"
        save_path_plots = save_path / "plots"

        id = str(EFFICIENCY) + "_" + str(i)


        results = pd.DataFrame(
            index=range(game.n_players + 1),
            data=np.vstack((model_calls_ratio, mse)).T,
            columns=("n_model_calls", "mse"),
        )
        results.to_csv(save_path / ("results_" +id))

        plt.figure()
        plt.plot(model_calls_ratio[plot_range] * 100, mse[plot_range], marker="o",label="GraphSHAP")
        plt.plot(model_calls_ratio[plot_range] * 100, mse_permutation[plot_range], marker="o",label="Permutation")
        plt.legend()
        plt.title("Explanation SSE" + str(x_graph))
        plt.savefig(save_path_plots / ("explanation_" + id))
        plt.show()

        plt.figure()
        plt.plot(model_calls_ratio[plot_range] * 100, np.log(mse[plot_range]), marker="o",label="GraphSHAP")
        plt.plot(model_calls_ratio[plot_range] * 100, np.log(mse_permutation[plot_range]), marker="o",label="Permutation")
        plt.legend()
        plt.title("Explanation SSE in log" + str(x_graph))
        plt.savefig(save_path_plots / ("explanation_log_" + id))
        plt.show()



        plt.figure()
        plt.plot(model_calls_ratio[plot_range] * 100, mse_moebius[plot_range], marker="o")
        plt.title("Möbius in log " + str(x_graph))
        plt.savefig(save_path_plots / ("moebius_log_" + id))
        plt.show()
