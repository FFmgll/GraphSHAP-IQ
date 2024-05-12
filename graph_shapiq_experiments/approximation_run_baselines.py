"""This module runs baseline approximation methods on the same settings as the GraphSHAP-IQ
approximations."""

import copy
import itertools
import os

from tqdm.auto import tqdm

from shapiq.explainer.graph import get_explanation_instances
from shapiq.approximator import (
    KernelSHAPIQ,
    SHAPIQ,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    PermutationSamplingSII,
)
from approximation_utils import (
    get_game_from_file_name,
    BASELINES_DIR,
    parse_file_name,
    save_interaction_value,
    create_results_overview_table,
)


def run_baseline(approx_name: str, budget: int, iteration: int, file_name: str, x_graph) -> None:
    """Run the baseline approximation on the given game.

    Args:
        approx_name: The name of the baseline approximation to run.
        budget: The budget to run the baseline approximation on.
        iteration: The iteration number of the baseline approximation.
        file_name: The file name of the game to run the baseline approximation on.

    Returns:
        The interaction values of the baseline approximation.
    """
    # get game from file name
    game = get_game_from_file_name(file_name, x_graph)
    game_settings = parse_file_name(file_name)

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
    # save the resulting InteractionValues
    save_interaction_value(
        interaction_values=interaction_values,
        game=game,
        model_id=game_settings["model_id"],
        dataset_name=game_settings["dataset_name"],
        n_layers=game_settings["n_layers"],
        save_exact=False,
        directory=os.path.join(BASELINES_DIR, approx_name),
        max_neighborhood_size=interaction_values.estimation_budget,
        efficiency=False,
        iteration=iteration,
        budget=budget,
    )


def approximate_baselines():
    """Runs the baseline approximations as specified in the configuration."""
    results_overview = create_results_overview_table()

    # check how many runs need to be done for the specified settings
    results_selection = results_overview[
        (results_overview["model_id"] == MODEL_ID)
        & (results_overview["dataset_name"] == DATASET_NAME)
        & (results_overview["n_layers"] == N_LAYERS)
        & (results_overview["small_graph"] == SMALL_GRAPH)
    ]

    # get all exact values that are not computed yet
    exact_selection = results_selection[results_selection["exact"] == True]
    exact_ids = exact_selection["instance_id"].tolist()
    print(f"Found {len(exact_ids)} instances matching this setting.")
    if not exact_ids:
        print(f"No instances found.")
        return

    # get all budgets for the instances
    budgets = {}
    for instance_id in exact_ids:
        budget_steps = results_selection[
            (results_selection["instance_id"] == instance_id)
            & (results_selection["approximation"] == "GraphSHAPIQ")
        ]["budget"].unique()
        budgets[instance_id] = copy.deepcopy(budget_steps)

    # get the dataset
    all_instances = get_explanation_instances(dataset_name=DATASET_NAME)

    # # list of tuples (approximator, instance_id, budget, iteration, file_name, x_graph)
    parameter_space, total_budget, unique_instances = [], 0, set()
    for approximator in APPROXIMATORS_TO_RUN:
        approx_selection = results_selection[
            (results_selection["approximation"] == approximator)
            & (results_selection["index"] == INDEX)
            & (results_selection["order"] == MAX_ORDER)
        ]
        for instance_id in exact_ids:
            setting = results_selection[results_selection["instance_id"] == instance_id].iloc[0]
            approximated = approx_selection[approx_selection["instance_id"] == instance_id]
            budget_steps = budgets[instance_id]
            n_evals_required = len(budget_steps) * ITERATIONS
            if approximated.shape[0] >= n_evals_required:
                continue
            for iteration, budget in itertools.product(range(1, ITERATIONS + 1), budget_steps):
                is_computed = approximated[
                    (approximated["iteration"] == iteration) & (approximated["budget"] == budget)
                ]
                if not is_computed.empty:
                    continue  # already computed
                if budget > MAX_APPROX_BUDGET:
                    continue
                file_name = setting["file_name"]
                x_graph = copy.deepcopy(all_instances[setting["data_id"]])
                parameter_space.append(
                    (approximator, instance_id, budget, iteration, file_name, x_graph)
                )
                total_budget += budget
                unique_instances.add(instance_id)

    if len(parameter_space) == 0:
        print(f"No instances to compute.")
        return
    print(
        f"Found {len(parameter_space)} instances to compute for {len(unique_instances)} unique "
        f"instances. Total budget: {total_budget}."
    )

    # run the baselines
    print(f"Approximating the baselines:", APPROXIMATORS_TO_RUN)
    with tqdm(
        total=total_budget, desc="Running the baseline approximations ", unit=" model calls"
    ) as pbar:
        for approximator, instance_id, budget, iteration, file_name, x_graph in parameter_space:
            run_baseline(approximator, budget, iteration, file_name, x_graph)
            pbar.update(budget)


if __name__ == "__main__":

    # game setup
    DATASET_NAME = "Mutagenicity"
    MODEL_ID = "GCN"
    N_LAYERS = 2

    ITERATIONS = 2

    INDEX = "k-SII"
    MAX_ORDER = 2

    MAX_APPROX_BUDGET = 10_000

    SMALL_GRAPH = True

    APPROXIMATORS_TO_RUN = [
        # KernelSHAPIQ.__name__,
        # SVARMIQ.__name__,
        PermutationSamplingSII.__name__,
    ]

    approximate_baselines()
