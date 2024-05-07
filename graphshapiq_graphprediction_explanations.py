from shapiq.explainer.graph.train_gnn import train_gnn
from torch_geometric.loader import DataLoader

# import modules
import torch
import pandas as pd

from shapiq import ExactComputer
from shapiq.explainer.graph import GraphSHAPIQ
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.explainer.graph import get_explanation_instances

import numpy as np
import os




def save_results(
    identifier,
    results,
    model_id,
    data_id,
    game,
    max_neighborhood_size,
    required_budget,
):
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
    all_interactions = (results["gt"] + results["GraphSHAPIQ"]).interaction_lookup
    values = np.zeros((len(all_interactions), 2))
    index = []

    for pos, (interaction, interaction_value) in enumerate(all_interactions.items()):
        values[pos, 0] = results["gt"][interaction]
        values[pos, 1] = results["GraphSHAPIQ"][interaction]
        index.append(interaction)

    results_pd = pd.DataFrame(index=index, data=values, columns=["gt", "GraphSHAPIQ"])
    results_pd["budget"] = required_budget

    save_name = "_".join(
        [
            identifier,
            model_id,
            str(data_id),
            str(game.n_players),
            str(max_neighborhood_size),
        ]
    )
    save_path = os.path.join("results/single_gt_instances", save_name + ".csv")
    results_pd.to_csv(save_path)


def explain_instances_with_gt(
    model_id, model, all_samples_to_explain, explanation_order=2, masking_mode="feature-removal"
):
    stop_processing = False
    counter = 0
    for data_id, x_graph in enumerate(all_samples_to_explain):
        if x_graph.num_nodes <= 16 and not stop_processing:
            print(x_graph.num_nodes)
            baseline = x_graph.x.mean(0)

            game = GraphGame(
                model,
                x_graph=x_graph,
                class_id=x_graph.y.item(),
                max_neighborhood_size=model.n_layers,
                masking_mode=masking_mode,
                normalize=True,
                baseline=baseline
            )
            # setup the explainer
            gSHAP = GraphSHAPIQ(game)

            results_moebius = {}
            results_interaction = {}

            exact_computer = ExactComputer(game.n_players, game)
            results_interaction["gt"] = exact_computer.shapley_interaction(
                index="k-SII", order=explanation_order
            )
            results_moebius["gt"] = exact_computer.moebius_transform()

            # Compute the Moebius values for all subsets
            results_moebius["GraphSHAPIQ"], results_interaction["GraphSHAPIQ"] = gSHAP.explain(
                max_interaction_size=gSHAP.max_size_neighbors,
                order=explanation_order,
                efficiency_routine=False,
            )

            required_budget = gSHAP.last_n_model_calls

            save_results(
                "gtmoebius",
                results_moebius,
                model_id,
                data_id,
                game,
                gSHAP.max_size_neighbors,
                required_budget,
            )
            save_results(
                "gtinteraction",
                results_interaction,
                model_id,
                data_id,
                game,
                gSHAP.max_size_neighbors,
                required_budget,
            )

            counter += 1
            if counter > 3:
                stop_processing = True




def explain_instances_no_gt(
    model_id, model, all_samples_to_explain, explanation_order=2, masking_mode="feature-removal"
):


    counter = 0
    stop_processing = False
    for data_id, x_graph in enumerate(all_samples_to_explain):
        if x_graph.num_nodes > 16 and x_graph.num_nodes <= 30 and not stop_processing:
            baseline = x_graph.x.mean(0)
            game = GraphGame(
                model,
                x_graph=x_graph,
                class_id=x_graph.y.item(),
                max_neighborhood_size=model.n_layers,
                masking_mode=masking_mode,
                normalize=True,
                baseline=baseline
            )
            # setup the explainer
            gSHAP = GraphSHAPIQ(game)

            if gSHAP.total_budget <= 2**16:
                print(x_graph.num_nodes)
                results_moebius = {}
                results_interaction = {}

                # Compute the Moebius values for all subsets
                results_moebius["GraphSHAPIQ"], results_interaction["GraphSHAPIQ"] = gSHAP.explain(
                    max_interaction_size=gSHAP.max_size_neighbors,
                    order=explanation_order,
                    efficiency_routine=False,
                )

                required_budget = gSHAP.last_n_model_calls
                results_interaction["gt"] = results_interaction["GraphSHAPIQ"]
                results_moebius["gt"] = results_moebius["GraphSHAPIQ"]

                save_results(
                    "largemoebius",
                    results_moebius,
                    model_id,
                    data_id,
                    game,
                    gSHAP.max_size_neighbors,
                    required_budget,
                )
                save_results(
                    "largeinteraction",
                    results_interaction,
                    model_id,
                    data_id,
                    game,
                    gSHAP.max_size_neighbors,
                    required_budget,
                )

                counter += 1
                if counter > 3:
                    stop_processing = True


if __name__ == "__main__":
    DATASET_NAMES = ["AIDS", "DHFR", "COX2", "BZR", "PROTEINS", "ENZYMES", "MUTAG", "Mutagenicity"]
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
                            explain_instances_no_gt(model_id, model, all_samples_to_explain)
