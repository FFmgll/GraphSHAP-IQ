"""This script runs the GraphSHAP-IQ approximation on different datasets and graphs."""

import copy


import torch

from tqdm.auto import tqdm

from utils_approximation import is_game_computed, save_interaction_value

from shapiq import InteractionValues
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.explainer.graph import (
    GraphSHAPIQ,
    _compute_baseline_value,
    get_explanation_instances,
    load_graph_model,
)


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
    for game in tqdm(games, desc="Running the GraphSHAP-IQ approximation"):
        moebius_values: dict[int, InteractionValues] = run_graph_shapiq_approximation(
            game, efficiency=efficiency
        )
        # save the resulting InteractionValues
        for size, values in moebius_values.items():
            save_interaction_value(
                interaction_values=values,
                game=game,
                model_id=MODEL_ID,
                dataset_name=DATASET_NAME,
                n_layers=N_LAYERS,
                max_neighborhood_size=size,
                efficiency=efficiency,
                save_exact=True,
            )


def run_graph_shapiq_approximation(
    game: GraphGame, efficiency: bool = True
) -> dict[int, InteractionValues]:
    """Run the GraphSHAP-IQ approximation on a given game for the specified neighbourhood sizes.

    Args:
        game: The game to run the approximation on.
        efficiency: Whether to use the efficiency routine for the approximation (True) or not
            (False).

    Returns:
        A dictionary mapping the neighbourhood sizes to the approximated möbius values with that
            neighbourhood size.
    """
    approximated_values = {}

    max_order = game.n_players
    approximator = GraphSHAPIQ(game)

    interaction_sizes = list(range(1, approximator.max_size_neighbors + 1))

    for interaction_size in interaction_sizes:
        moebius, _ = approximator.explain(
            max_interaction_size=interaction_size,
            order=max_order,
            efficiency_routine=efficiency,
        )
        budget_used = approximator.last_n_model_calls
        moebius.estimation_budget = budget_used
        moebius.estimated = False if interaction_size == approximator.max_size_neighbors else True
        moebius.sparsify(threshold=1e-8)
        approximated_values[interaction_size] = copy.deepcopy(moebius)

    return approximated_values


if __name__ == "__main__":

    # run setup
    N_GAMES = 2
    MAX_N_PLAYERS = 20

    MODEL_ID = "GCN"  # one of GCN GIN
    DATASET_NAME = "Mutagenicity"  # one of MUTAG PROTEINS ENZYMES AIDS DHFR COX2 BZR Mutagenicity
    N_LAYERS = 1  # one of 1 2 3 4
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
