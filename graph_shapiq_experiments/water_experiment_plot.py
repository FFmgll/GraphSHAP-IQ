"""This script plots the water use case experiment's explanation graphs."""

import os
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from shapiq.interaction_values import InteractionValues
from shapiq.explainer.graph import _compute_baseline_value
from shapiq.explainer.graph.load_water_quality import load_quality_model, load_water_quality_data
from shapiq.games.benchmark.local_xai.benchmark_graph import GraphGame
from shapiq.plot.explanation_graph import explanation_graph_plot
from shapiq.moebius_converter import MoebiusConverter

WATER_SAVE_DIR = "water_plots"
os.makedirs(WATER_SAVE_DIR, exist_ok=True)

if __name__ == "__main__":

    INDEX = "k-SII"
    MAX_ORDER = 2

    TIME_STEPS = [0]  # list(range(1, 62))
    file_names_to_find = [f"{t}_graphshapiq.interaction_values" for t in TIME_STEPS]

    # load the model and dataset
    model = load_quality_model()
    model.eval()
    ds_test = load_water_quality_data(batch_size=1)["test"]
    all_instances = [graph for graph in ds_test]

    all_interactions = {}
    for file in os.listdir(WATER_SAVE_DIR):
        if file not in file_names_to_find:
            continue
        time_step = int(file.split("_")[0])
        values = InteractionValues.load(os.path.join(WATER_SAVE_DIR, file))
        index_values = MoebiusConverter(values)(index=INDEX, order=MAX_ORDER)
        all_interactions[time_step] = index_values

    # get the max and min absolute interaction value in all_interactions
    max_abs = max([np.max(np.abs(interaction.values)) for interaction in all_interactions.values()])
    min_abs = min([np.min(np.abs(interaction.values)) for interaction in all_interactions.values()])

    # load a single graph for the plot and positions
    ds_test = load_water_quality_data(batch_size=1)["test"]
    pos = ds_test.dataset.node_pos
    edge_index = ds_test.dataset.edge_index
    graph = nx.Graph(edge_index.T.tolist())

    # do the plotting for all time steps
    for time_step in TIME_STEPS:
        try:
            interaction = all_interactions[time_step]
        except KeyError:
            print(f"Skipping time step {time_step} because no interaction values are computed.")
            continue
        # run the model and get the prediction
        x_graph = all_instances[time_step]
        with torch.no_grad():
            predicted_chlorination = model(
                x_graph.x, x_graph.edge_index, x_graph.edge_features, x_graph.batch
            )
        test_loss = F.l1_loss(x_graph.label, predicted_chlorination).cpu().numpy().item()

        # create a game to get the baseline value (output of not-normalized game)
        game = GraphGame(
            model,
            x_graph=x_graph,
            class_id=0,
            max_neighborhood_size=model.n_layers,
            masking_mode="feature-removal",
            normalize=True,
            baseline=_compute_baseline_value(x_graph),
            instance_id=int(time_step),
        )
        baseline_value = game.normalization_value

        print(interaction)
        interactions_sum = interaction.values.sum()
        sum_of_interactions = interactions_sum + baseline_value
        print("Summary of Interaction Values at time step", time_step)
        print("Sum:", sum_of_interactions)
        print("Mean:", np.abs(interaction.values).mean())
        print("Std:", np.abs(interaction.values).std())
        print("Max:", np.abs(interaction.values).max())
        print("Min:", np.abs(interaction.values).min())
        explanation_graph_plot(
            interaction,
            graph=graph,
            min_max_interactions=(min_abs, max_abs),
            pos=pos,
            n_interactions=100,
            random_seed=200,
            compactness=10,
            size_factor=20,
            node_size_scaling=0.5,
        )
        index_title = str(MAX_ORDER) + "-SII" if INDEX == "k-SII" else INDEX
        plt.title(
            f"{index_title} Explanation Graph at Time Step {time_step}\n"
            f"Sum of Interaction Values: {sum_of_interactions:.3f} = {interactions_sum:.3f} + "
            f"{baseline_value:.3f} (interactions + baseline)\n"
            f"Predicted Chlorination: {predicted_chlorination.item():.3f}, MAE: {test_loss:.3f}"
        )
        plt.show()
        print("\n")
