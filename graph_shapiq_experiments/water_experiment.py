"""This script runs the water use case experiment."""

import os
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

from shapiq.interaction_values import InteractionValues
from shapiq.approximator import KernelSHAPIQ
from shapiq.explainer.graph import _compute_baseline_value, GraphSHAPIQ
from shapiq.explainer.graph.load_water_quality import load_quality_model, load_water_quality_data
from shapiq.games.benchmark.local_xai.benchmark_graph import GraphGame
from shapiq.plot.explanation_graph import explanation_graph_plot
from shapiq.moebius_converter import MoebiusConverter

WATER_SAVE_DIR = "water_plots"
os.makedirs(WATER_SAVE_DIR, exist_ok=True)

if __name__ == "__main__":

    PLOT_GRAPH = True  # plot the graph from the dataset
    USE_GRAPH_SHAPIQ = True  # if False, KernelSHAPIQ is used instead of GraphSHAPIQ
    LOAD_FROM_DISK = False  # you can turn this to True then it will load the values from disk

    TIMESTEP = 20
    SAVE_PREFIX = f"{TIMESTEP}_"

    # load the model
    model = load_quality_model()
    model.eval()

    # get the dataset/data point
    ds_test = load_water_quality_data(batch_size=1)["test"]
    all_instances = [graph for graph in ds_test]
    x_graph = all_instances[TIMESTEP]

    # Sample a batch from dataset
    # Call the model prediction.shape: [BatchSize, 1]
    with torch.no_grad():
        predicted_chlorination = model(
            x_graph.x, x_graph.edge_index, x_graph.edge_features, x_graph.batch
        )
    test_loss = F.l1_loss(x_graph.label, predicted_chlorination).cpu().numpy().item()
    print(f"Model achieves {test_loss:.4f} MAE on the test set.")
    print(f"Predicted Chlorination at Time Step {TIMESTEP}: ", predicted_chlorination)

    if PLOT_GRAPH:
        # adapted code from Luca for plotting the graph
        # To access node positions and plot a graph:
        pos = ds_test.dataset.node_pos
        edge_index = ds_test.dataset.edge_index
        G = nx.Graph(edge_index.T.tolist())
        n_nodes = G.number_of_nodes()
        # To draw the graph: (Here node color corresponds to chlorine values of the 5th sample in the batch)
        X = x_graph.x.reshape(-1).cpu().numpy()
        nx.draw(G, pos=pos, node_color=plt.cm.viridis(X)[G.nodes])
        plt.savefig(os.path.join(WATER_SAVE_DIR, SAVE_PREFIX + "concentration.pdf"))
        plt.show()

    # get the graph for the explanation later on
    G = nx.Graph(x_graph.edge_index.T.tolist())
    pos = ds_test.dataset.node_pos

    # initialize the Game
    baseline = _compute_baseline_value(x_graph)
    game = GraphGame(
        model,
        x_graph=x_graph,
        class_id=0,
        max_neighborhood_size=model.n_layers,
        masking_mode="feature-removal",
        normalize=True,
        baseline=baseline,
        instance_id=int(TIMESTEP),
    )

    if USE_GRAPH_SHAPIQ and not LOAD_FROM_DISK:
        print("Using GraphSHAPIQ...")
        # use graph shapiq
        explainer = GraphSHAPIQ(game)
        moebius_values, _ = explainer.explain()
        moebius_values.save(
            os.path.join(WATER_SAVE_DIR, SAVE_PREFIX + "graphshapiq.interaction_values")
        )
        print(moebius_values)
        result = MoebiusConverter(moebius_values)(index="k-SII", order=2)
    elif not USE_GRAPH_SHAPIQ and not LOAD_FROM_DISK:
        print("Using KernelSHAPIQ...")
        # use the kernel approximator
        approximator = KernelSHAPIQ(n=game.n_players, index="k-SII", max_order=2)
        result = approximator.approximate(game=game, budget=15_000)
        result.save(os.path.join(WATER_SAVE_DIR, SAVE_PREFIX + "kernelshapiq.interaction_values"))
    else:
        # load from disk
        if USE_GRAPH_SHAPIQ:
            print("Loading GraphSHAPIQ from disk...")
            result = InteractionValues.load(
                os.path.join(WATER_SAVE_DIR, SAVE_PREFIX + "graphshapiq.interaction_values")
            )
        else:
            print("Loading KernelSHAPIQ from disk...")
            result = InteractionValues.load(
                os.path.join(WATER_SAVE_DIR, SAVE_PREFIX + "kernelshapiq.interaction_values")
            )
    print(result)

    if result.index == "Moebius":
        result = MoebiusConverter(result)(index="k-SII", order=2)

    fig, axis = explanation_graph_plot(
        interaction_values=result,
        graph=G,
        n_interactions=100,
        random_seed=200,
        compactness=10,
        pos=pos,
        size_factor=4,
        node_size_scaling=0.5,
    )
    plt.savefig(os.path.join(WATER_SAVE_DIR, SAVE_PREFIX + "explanation.pdf"))
    plt.show()
