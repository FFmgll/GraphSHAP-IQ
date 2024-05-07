""" This script runs the GraphSHAP-IQ approximation on different datasets and graphs."""

from shapiq.explainer.graph.train_gnn import train_gnn
from torch_geometric.loader import DataLoader

# import modules
import torch
import pandas as pd
import numpy as np
import os

from shapiq.explainer.graph import GraphSHAPIQ
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.explainer.graph import _compute_baseline_value, get_explanation_instances

def save_results(model_id, data_id, game, sse, max_neighborhood_size, gshap_budget):
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

    final_results["budget"] = gshap_budget

    save_name = "_".join(
        [
            model_id,
            str(data_id),
            str(game.n_players),
            str(max_neighborhood_size),
        ]
    )

    save_path = os.path.join("results/approximation", save_name + ".csv")
    final_results.to_csv(save_path)


def explain_instances(
    model_id,
    model,
    all_samples_to_explain,
    efficiency_mode,
    explanation_order=2,
    masking_mode="feature-removal",
):
    # Stop processing after counter reaches a certain limit, see below
    stop_processing = False
    counter = 0
    print("Running without GT...", model_id)
    for data_id, x_graph in enumerate(all_samples_to_explain):
        if not stop_processing:
            baseline = _compute_baseline_value(x_graph)
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
            if gSHAP.total_budget <= 2**12 and gSHAP.max_size_neighbors <= 12:
                # print(x_graph.num_nodes, gSHAP.max_size_neighbors)

                sse = {}
                gshap_moebius = {}
                gshap_interactions = {}
                gshap_budget = {}

                approx_id_interaction = "GraphSHAPIQ_" + str(efficiency_mode) + "_interaction"
                approx_id_moebius = "GraphSHAPIQ_" + str(efficiency_mode) + "_moebius"

                sse[approx_id_interaction] = {}
                sse[approx_id_moebius] = {}
                for max_interaction_size in range(1, gSHAP.max_size_neighbors + 1):
                    # Compute the Moebius values for all subsets
                    (
                        gshap_moebius[max_interaction_size],
                        gshap_interactions[max_interaction_size],
                    ) = gSHAP.explain(
                        max_interaction_size=max_interaction_size,
                        order=explanation_order,
                        efficiency_routine=efficiency_mode,
                    )
                    gshap_budget[
                        max_interaction_size
                    ] = gSHAP.last_n_model_calls

                # Set ground-truth to
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

                save_results(
                    model_id, data_id, game, sse, gSHAP.max_size_neighbors, gshap_budget
                )
                counter += 1
        if counter >= 20:
            stop_processing = True


if __name__ == "__main__":
    DATASET_NAMES = [ "MUTAG","PROTEINS", "ENZYMES", "AIDS", "DHFR", "COX2", "BZR", "Mutagenicity"]
    MODEL_TYPES = ["GCN"]
    N_LAYERS = [1, 2, 3, 4]
    NODE_BIASES = [True]  # [False,True]
    GRAPH_BIASES = [True]  # [False,True]
    EFFICIENCY_MODES = [True] # [False,True]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in DATASET_NAMES:
        all_samples_to_explain = get_explanation_instances(dataset_name)
        for model_type in MODEL_TYPES:
            for n_layers in N_LAYERS:
                for efficiency_mode in EFFICIENCY_MODES:
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
                            explain_instances(model_id,
                                              model,
                                              all_samples_to_explain,
                                              efficiency_mode)
