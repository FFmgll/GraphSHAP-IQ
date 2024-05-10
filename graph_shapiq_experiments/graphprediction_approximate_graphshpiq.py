"""This script runs the GraphSHAP-IQ approximation on different datasets and graphs."""

import copy
import os
from typing import Optional

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


def save_interaction_value(
    interaction_values: InteractionValues,
    game: GraphGame,
    max_neighborhood_size: int,
    efficiency: bool,
) -> None:
    """Save the interaction values to a file.

    The naming convention for the file is the following attributes separated by underscore:
        - Type of model, e.g. GCN, GIN
        - Dataset name, e.g. MUTAG
        - Number of Graph Convolutions, e.g. 2 graph conv layers
        - Data ID: a technical identifier of the explained instance
        - Number of players, i.e. number of nodes in the graph
        - Maximum neighbourhood size as integer
        - Efficiency routine used for the approximation as boolean


    Args:
        interaction_values: The interaction values to save.
        game: The game for which the interaction values were computed.
        max_neighborhood_size: The maximum neighbourhood size for which the interaction values were
            computed.
        efficiency: Whether the efficiency routine was used for the approximation (True) or not
            (False).
    """
    save_name = (
        "_".join(
            [
                str(MODEL_ID),
                str(DATASET_NAME),
                str(N_LAYERS),
                str(game.game_id),
                str(game.n_players),
                str(max_neighborhood_size),
                str(efficiency),
            ]
        )
        + ".interaction_values"
    )
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)
    save_path = os.path.join(STORAGE_DIR, save_name)
    interaction_values.save(save_path)


def run_graph_shapiq_approximations(
    games: list[GraphGame],
    efficiency: bool = True,
) -> None:
    """Run the GraphSHAP-IQ approximation on a list of games and save the results.

    Args:
        games: The games to run the approximation on.
        efficiency: Whether to use the efficiency routine for the approximation (True) or not
            (False).
    """

    # run the approximation
    results: list[dict[int, InteractionValues]] = []
    for game in tqdm(games, desc="Running the GraphSHAP-IQ approximation"):
        results.append(run_graph_shapiq_approximation(game, efficiency=efficiency))

    # save the resulting InteractionValues
    for moebius_values, game in zip(results, games):
        for size, values in moebius_values.items():
            save_interaction_value(values, game, size, efficiency)


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


def is_game_computed(
    model_type: str, dataset_name: str, n_layers: int, data_id: int, efficiency: bool
) -> bool:
    """Check whether the game has already been computed."""
    all_files = os.listdir(STORAGE_DIR)
    for file in all_files:
        split_file = file.split(".")[0].split("_")
        if (
            split_file[0] == model_type
            and split_file[1] == dataset_name
            and split_file[2] == str(n_layers)
            and split_file[3] == str(data_id)
            and split_file[6] == str(efficiency)
        ):
            return True
    return False


if __name__ == "__main__":

    # run setup
    N_GAMES = 1000
    MAX_N_PLAYERS = 16

    MODEL_ID = "GCN"  # one of GCN GIN
    DATASET_NAME = "Mutagenicity"  # one of MUTAG PROTEINS ENZYMES AIDS DHFR COX2 BZR Mutagenicity
    N_LAYERS = 3  # one of 1 2 3 4
    EFFICIENCY_MODE = True  # one of True False

    # see whether a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the model for the approximation
    model = load_graph_model(MODEL_ID, DATASET_NAME, N_LAYERS, device=device)

    # set the games up for the approximation
    games_to_run = []
    explanation_instances = get_explanation_instances(DATASET_NAME)
    for data_id, x_graph in enumerate(explanation_instances):
        if is_game_computed(MODEL_ID, DATASET_NAME, N_LAYERS, data_id, EFFICIENCY_MODE):
            continue
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
        if game_to_run.n_players <= MAX_N_PLAYERS:
            games_to_run.append(game_to_run)
        if len(games_to_run) >= N_GAMES:
            break

    print(f"Running the GraphSHAP-IQ approximation on {len(games_to_run)} games.")

    # run the approximation
    run_graph_shapiq_approximations(games_to_run, efficiency=EFFICIENCY_MODE)
