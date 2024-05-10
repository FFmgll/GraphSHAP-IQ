"""This module runs basline approximation methods on the same settings as the GraphSHAP-IQ
approximations."""

import os

import torch
from tqdm.auto import tqdm

from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.interaction_values import InteractionValues
from shapiq.approximator import (
    KernelSHAPIQ,
    SHAPIQ,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    PermutationSamplingSII,
)
from utils_approximation import (
    load_exact_values_from_disk,
    get_games_from_file_names,
    load_approximation_values_from_disk,
    GRAPHSHAPIQ_APPROXIMATION_DIR,
    BASELINES_DIR,
    save_interaction_value,
    is_file_computed,
)


def run_baseline(game: GraphGame, approx_name: str, budget: int) -> InteractionValues:
    """Run the baseline approximation on the given game.

    Args:
        game: The game to run the baseline approximation on.
        approx_name: The name of the baseline approximation to run.
        budget: The budget to run the baseline approximation on.

    Returns:
        The interaction values of the baseline approximation.
    """
    n_players = game.n_players
    if approx_name == KernelSHAPIQ.__name__:
        approximator = KernelSHAPIQ(n=n_players, index=INDEX, max_order=MAX_ORDER)
    elif approx_name == InconsistentKernelSHAPIQ.__name__:
        approximator = InconsistentKernelSHAPIQ(n=n_players, index=INDEX, max_order=MAX_ORDER)
    elif approx_name == SHAPIQ.__name__:
        approximator = SHAPIQ(n=n_players, index=INDEX, max_order=MAX_ORDER)
    elif approx_name == SVARMIQ.__name__:
        approximator = SVARMIQ(n=n_players, index=INDEX, max_order=MAX_ORDER)
    elif approx_name == PermutationSamplingSII.__name__:
        approximator = PermutationSamplingSII(n=n_players, index=INDEX, max_order=MAX_ORDER)
    else:
        raise ValueError(f"Approximator {approx_name} not found.")
    interaction_values = approximator.approximate(budget=budget, game=game)
    return interaction_values


def run_approximators_on_graph_games(
    games_to_run: list[GraphGame], game_budget_steps: list[list[int]]
) -> None:
    """Run the approximators on the graph games.

    Args:
        games_to_run: The list of graph games to run the approximators on.
        game_budget_steps: The list of budget steps to run the approximators on for each game.
    """
    parameter_space = []
    total_budget = 0
    for game_index, game in enumerate(games_to_run):
        for approx_name in APPROXIMATORS_TO_RUN:
            for budget in game_budget_steps[game_index]:
                for iteration in range(1, ITERATIONS + 1):
                    total_budget += budget
                    parameter_space.append((game, approx_name, budget, iteration))

    with tqdm(total=total_budget, desc="Running the baseline approximations") as pbar:
        for game, approx_name, budget, iteration in parameter_space:
            interaction_values = run_baseline(game, approx_name, budget)
            save_dir = os.path.join(BASELINES_DIR, approx_name)
            # save the resulting InteractionValues
            save_interaction_value(
                interaction_values=interaction_values,
                game=game,
                model_id=MODEL_ID,
                dataset_name=DATASET_NAME,
                n_layers=N_LAYERS,
                save_exact=False,
                directory=save_dir,
                max_neighborhood_size=interaction_values.estimation_budget,
                efficiency=False,
                iteration=iteration,
            )
            pbar.update(budget)


if __name__ == "__main__":

    # game setup
    DATASET_NAME = "Mutagenicity"
    MODEL_ID = "GCN"
    N_LAYERS = 2

    MAX_GAMES = 3
    ITERATIONS = 2

    INDEX = "k-SII"
    MAX_ORDER = 2

    APPROXIMATORS_TO_RUN = [
        KernelSHAPIQ.__name__,
        SVARMIQ.__name__,
        PermutationSamplingSII.__name__,
    ]

    # get the games from the available exact interaction_values
    exact_values, file_names = load_exact_values_from_disk(
        model_id=MODEL_ID, dataset_name=DATASET_NAME, n_layers=N_LAYERS
    )

    # drop the file names that are already computed
    file_names = [
        file_name for file_name in file_names if not is_file_computed(file_name, BASELINES_DIR)
    ]

    # get the games from the file names
    games = get_games_from_file_names(file_names)
    games, file_names = games[:MAX_GAMES], file_names[:MAX_GAMES]

    # get the budget steps for each game
    budget_steps = []
    for file_name in file_names:
        graph_shap_iq_approximations = load_approximation_values_from_disk(
            exact_file_name=file_name, directory=GRAPHSHAPIQ_APPROXIMATION_DIR
        )
        budget_per_game = []
        for approximation in graph_shap_iq_approximations:
            budget_step = approximation.estimation_budget
            if budget_step not in budget_steps:
                budget_per_game.append(budget_step)
        budget_steps.append(budget_per_game)

    print(f"Running the baseline approximations for {len(games)} games.")

    # run the approximators on the graph games
    run_approximators_on_graph_games(games, budget_steps)
