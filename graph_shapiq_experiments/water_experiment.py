"""This script runs the water use case experiment."""

import os
import torch
import torch.nn.functional as F
import networkx as nx

from shapiq.interaction_values import InteractionValues
from shapiq.approximator import KernelSHAPIQ
from shapiq.explainer.graph import _compute_baseline_value, GraphSHAPIQ
from shapiq.explainer.graph.load_water_quality import load_quality_model, load_water_quality_data
from shapiq.games.benchmark.local_xai.benchmark_graph import GraphGame
from shapiq.moebius_converter import MoebiusConverter

WATER_SAVE_DIR = "water_plots"
os.makedirs(WATER_SAVE_DIR, exist_ok=True)

if __name__ == "__main__":

    USE_GRAPH_SHAPIQ = True  # if False, KernelSHAPIQ is used instead of GraphSHAPIQ

    SAVE_PREFIX = "EMPTY_"

    TIME_STEPS = [0]  # list(range(11, 20))

    for TIME_STEP in TIME_STEPS:

        SAVE_PREFIX = SAVE_PREFIX + f"{TIME_STEP}_"

        # load the model
        model = load_quality_model()
        model.eval()

        # get the dataset/data point
        ds_test = load_water_quality_data(batch_size=1)["test"]
        all_instances = [graph for graph in ds_test]
        x_graph = all_instances[TIME_STEP]

        # Sample a batch from dataset
        # Call the model prediction.shape: [BatchSize, 1]
        with torch.no_grad():
            predicted_chlorination = model(
                x_graph.x, x_graph.edge_index, x_graph.edge_features, x_graph.batch
            )
        test_loss = F.l1_loss(x_graph.label, predicted_chlorination).cpu().numpy().item()
        print(f"Model achieves {test_loss:.4f} MAE on the test set.")
        print(f"Predicted Chlorination at Time Step {TIME_STEP}: ", predicted_chlorination)

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
            instance_id=int(TIME_STEP),
        )

        if USE_GRAPH_SHAPIQ:
            print("Using GraphSHAPIQ...")
            # use graph shapiq
            explainer = GraphSHAPIQ(game)
            moebius_values, _ = explainer.explain()
            moebius_values.save(
                os.path.join(WATER_SAVE_DIR, SAVE_PREFIX + "graphshapiq.interaction_values")
            )
            print(moebius_values)
            result = MoebiusConverter(moebius_values)(index="k-SII", order=2)
        elif not USE_GRAPH_SHAPIQ:
            print("Using KernelSHAPIQ...")
            # use the kernel approximator
            approximator = KernelSHAPIQ(n=game.n_players, index="k-SII", max_order=2)
            result = approximator.approximate(game=game, budget=15_000)
            result.save(
                os.path.join(WATER_SAVE_DIR, SAVE_PREFIX + "kernelshapiq.interaction_values")
            )
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
