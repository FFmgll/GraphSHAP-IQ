"""This module runs basline approximation methods on the same settings as the GraphSHAP-IQ
approximations."""

import copy
import multiprocessing as mp
import os
from typing import Optional

from tqdm.auto import tqdm
import torch

from shapiq.explainer.graph import (
    _compute_baseline_value,
    get_explanation_instances,
    load_graph_model,
)
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.interaction_values import InteractionValues
from shapiq.games.benchmark.run import run_benchmark, save_results
from shapiq.approximator import (
    KernelSHAPIQ,
    SHAPIQ,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    PermutationSamplingSII,
)

# TODO fix the MSE and other metrics in benchmark to work with potentially None values by dividing by the number of n_interactions.
STORAGE_DIR = os.path.join("..", "results", "approximation", "graphshapiq")


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
    games: list[GraphGame], budget_steps: list[list[int]], n_jobs: int = 1
) -> None:
    """Run the approximators on the graph games.

    Args:
        games: The list of graph games to run the approximators on.
        budget_steps: The list of budget steps to run the approximators on for each game.
        n_jobs: The number of parallel jobs to run. Defaults to 1.
    """
    parameter_space = []
    for game_index, game in enumerate(games):
        for approx_name in APPROXIMATORS_TO_RUN:
            for budget in budget_steps[game_index]:
                parameter_space.append((game, approx_name, budget))

    with mp.Pool(n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(run_baseline, parameter_space),
                total=len(parameter_space),
                desc="Running benchmark:",
                unit=" experiments",
            )
        )

    # TODO: finish the implementation of this function


if __name__ == "__main__":

    INDEX = "k-SII"
    MAX_ORDER = 2

    APPROXIMATORS_TO_RUN = [
        KernelSHAPIQ.__name__,
        InconsistentKernelSHAPIQ.__name__,
        SHAPIQ.__name__,
        SVARMIQ.__name__,
        PermutationSamplingSII.__name__,
    ]

    # TODO load the games and derive the budget steps from the games

    run_approximators_on_graph_games(games_to_run, budget_steps, n_jobs=6)
