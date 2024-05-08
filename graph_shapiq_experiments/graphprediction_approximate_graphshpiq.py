"""This script runs the GraphSHAP-IQ approximation on different datasets and graphs."""

import copy
import os
from typing import Optional
import multiprocessing as mp

import torch

from tqdm.auto import tqdm

from shapiq import InteractionValues
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.explainer.graph import (
    GraphSHAPIQ,
    _compute_baseline_value,
    get_explanation_instances,
    load_graph_model,
)


STORAGE_DIR = os.path.join("..", "results", "approximation", "graphshapiq")


def save_interaction_value(interaction_values: InteractionValues, game: GraphGame) -> None:
    """Save the interaction values to a file.

    The naming convention for the file is the following attributes separated by underscore:
        - Type of model, e.g. GCN, GIN
        - Dataset name, e.g. MUTAG
        - Number of Graph Convolutions, e.g. 2 graph conv layers
        - Graph bias, e.g. True, if the linear layer after global pooling has a bias
        - Node bias, e.g. True, if the convolution layers have a bias
        - Data ID: a technical identifier of the explained instance
        - Number of players, i.e. number of nodes in the graph
        - Largest neighborhood size as integer


    Args:
        interaction_values: The interaction values to save.
        game: The game for which the interaction values were computed.
    """
    save_name = "_".join(
        [
            str(MODEL_ID),
            str(DATASET_NAME),
            str(N_LAYERS),
            str(GRAPH_BIAS),
            str(NODE_BIAS),
            str(game.game_id),
            str(game.n_players),
            str(game.max_neighborhood_size),
        ]
    )
    save_path = os.path.join(STORAGE_DIR, save_name)
    interaction_values.save(save_path)


def run_graph_shapiq_approximations(
    games: list[GraphGame],
    efficiency: bool = True,
    n_jobs: int = 1,
) -> None:
    """Run the GraphSHAP-IQ approximation on a list of games and save the results.

    Args:
        games: The games to run the approximation on.
        efficiency: Whether to use the efficiency routine for the approximation (True) or not
            (False).
        n_jobs: The number of parallel jobs to run.
    """

    # run the approximation
    parameter_space = [(game, efficiency) for game in games]
    with mp.Pool(n_jobs) as pool:
        results: list[dict[int, InteractionValues]] = list(
            tqdm(
                pool.imap_unordered(run_graph_shapiq_approximation, parameter_space),
                total=len(parameter_space),
                desc="Running GraphSHAP-IQ:",
                unit=" experiments",
            )
        )

    # save the resulting InteractionValues
    for moebius_values, game in zip(results, games):
        for size, values in moebius_values.items():
            save_interaction_value(values, game)


def run_graph_shapiq_approximation(
    game: GraphGame, efficiency: bool = True, neighbourhood_size: Optional[list[int]] = None
) -> dict[int, InteractionValues]:
    """Run the GraphSHAP-IQ approximation on a given game for the specified neighbourhood sizes.

    Args:
        game: The game to run the approximation on.
        efficiency: Whether to use the efficiency routine for the approximation (True) or not
            (False).
        neighbourhood_size: The maximum neighbourhood sizes to consider for the approximation.
            Defaults to `None`, which uses all neighbourhood sizes up to the maximum neighbourhood
            size of the game.

    Returns:
        A dictionary mapping the neighbourhood sizes to the approximated möbius values with that
            neighbourhood size.
    """
    approximated_values = {}
    if neighbourhood_size is None:
        neighbourhood_size = list(range(1, game.max_neighborhood_size + 1))

    max_order = game.n_players
    approximator = GraphSHAPIQ(game)

    for size in neighbourhood_size:
        moebius, _ = approximator.explain(
            max_interaction_size=size,
            order=max_order,
            efficiency_routine=efficiency,
        )
        budget_used = approximator.last_n_model_calls
        moebius.estimation_budget = budget_used
        moebius.estimated = True if size < game.max_neighborhood_size else False
        moebius.sparsify(threshold=1e-5)
        approximated_values[size] = copy.deepcopy(moebius)

    return approximated_values


if __name__ == "__main__":

    MODEL_ID = "GCN"  # one of GCN GIN
    DATASET_NAME = "MUTAG"  # one of MUTAG PROTEINS ENZYMES AIDS DHFR COX2 BZR Mutagenicity
    N_LAYERS = 2  # one of 1 2 3 4
    NODE_BIAS = True  # one of True False
    GRAPH_BIAS = True  # one of True False
    EFFICIENCY_MODE = True  # one of True False

    # see whether a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the model for the approximation
    model = load_graph_model(MODEL_ID, DATASET_NAME, N_LAYERS, NODE_BIAS, GRAPH_BIAS, device=device)

    # set the games up for the approximation
    games_to_run = []
    explanation_instances = get_explanation_instances(DATASET_NAME)
    for data_id, x_graph in enumerate(explanation_instances.items()):
        baseline = _compute_baseline_value(x_graph)
        game_to_run = GraphGame(
            model,
            x_graph=x_graph,
            class_id=x_graph.y.item(),
            max_neighborhood_size=model.n_layers,
            masking_mode="feature-removal",
            normalize=True,
            baseline=baseline,
            instance_id=int(data_id),
        )
        games_to_run.append(game_to_run)

    # run the approximation
    run_graph_shapiq_approximations(games_to_run, efficiency=EFFICIENCY_MODE, n_jobs=1)
