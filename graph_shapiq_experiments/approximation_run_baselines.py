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
    PermutationSamplingSV,
    KernelSHAP,
)
from approximation_utils import (
    get_game_from_file_name,
    BASELINES_DIR,
    parse_file_name,
    save_interaction_value,
    create_results_overview_table,
)


def run_baseline(
    approx_name: str,
    budget: int,
    iteration: int,
    file_name: str,
    x_graph,
    index: str,
    max_order: int,
    interaction_size,
) -> None:
    """Run the baseline approximation on the given game.

    Args:
        approx_name: The name of the baseline approximation to run.
        budget: The budget to run the baseline approximation on.
        iteration: The iteration number of the baseline approximation.
        file_name: The file name of the game to run the baseline approximation on.
        x_graph: The graph instance to run.
        index: The index for which the approximator is to be run.
        max_order: The maximum approximation order.
        interaction_size: The interaction size of the graph-shapiq approximation.

    Returns:
        The interaction values of the baseline approximation.
    """
    # get game from file name
    game = get_game_from_file_name(file_name, x_graph)
    game_settings = parse_file_name(file_name)

    n_players = game.n_players
    if approx_name == KernelSHAPIQ.__name__:
        approximator = KernelSHAPIQ(n=n_players, index=index, max_order=max_order)
    elif approx_name == InconsistentKernelSHAPIQ.__name__:
        approximator = InconsistentKernelSHAPIQ(n=n_players, index=index, max_order=max_order)
    elif approx_name == SHAPIQ.__name__:
        approximator = SHAPIQ(n=n_players, index=index, max_order=max_order)
    elif approx_name == SVARMIQ.__name__:
        approximator = SVARMIQ(n=n_players, index=index, max_order=max_order)
    elif approx_name == PermutationSamplingSII.__name__:
        approximator = PermutationSamplingSII(n=n_players, index=index, max_order=max_order)
    elif approx_name == PermutationSamplingSV.__name__ and index == "SV" and max_order == 1:
        approximator = PermutationSamplingSV(n=n_players)
    elif approx_name == KernelSHAP.__name__ and index == "SV" and max_order == 1:
        approximator = KernelSHAP(n=n_players)
    else:
        raise ValueError(f"Approximator {approx_name} not found. Maybe wrong settings?")
    interaction_values = approximator.approximate(budget=budget, game=game)
    # save the resulting InteractionValues
    save_interaction_value(
        interaction_values=interaction_values,
        game=game,
        model_id=game_settings["model_id"],
        dataset_name=game_settings["dataset_name"],
        n_layers=game_settings["n_layers"],
        save_exact=False,
        save_directory=os.path.join(BASELINES_DIR, approx_name),
        max_interaction_size=interaction_size,
        efficiency=False,
        iteration=iteration,
        budget=budget,
    )


def approximate_baselines(
    *,
    model_id,
    dataset_name,
    n_layers,
    small_graph,
    iterations,
    approximators_to_run,
    index,
    max_order,
    max_approx_budget,
) -> None:
    """Runs the baseline approximations as specified in the configuration."""
    results_overview = create_results_overview_table()

    # check how many runs need to be done for the specified settings
    results_selection = results_overview[
        (results_overview["model_id"] == model_id)
        & (results_overview["dataset_name"] == dataset_name)
        & (results_overview["n_layers"] == n_layers)
        & (results_overview["small_graph"] == small_graph)
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
    all_instances = get_explanation_instances(dataset_name=dataset_name)

    # # list of tuples (approximator, instance_id, budget, iteration, file_name, x_graph)
    parameter_space, total_budget, unique_instances = [], 0, set()
    for approximator in approximators_to_run:
        approx_selection = results_selection[
            (results_selection["approximation"] == approximator)
            & (results_selection["index"] == index)
            & (results_selection["order"] == max_order)
        ]
        for instance_id in exact_ids:
            setting = results_selection[results_selection["instance_id"] == instance_id].iloc[0]
            approximated = approx_selection[approx_selection["instance_id"] == instance_id]
            budget_steps = budgets[instance_id]
            n_evals_required = len(budget_steps) * iterations
            if approximated.shape[0] >= n_evals_required:
                continue
            for iteration, budget in itertools.product(range(1, iterations + 1), budget_steps):
                is_computed = approximated[
                    (approximated["iteration"] == iteration) & (approximated["budget"] == budget)
                ]
                if not is_computed.empty:
                    continue  # already computed
                if budget > max_approx_budget:
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
    print(f"Approximating the baselines:", approximators_to_run)
    print(
        f"Settings: max_budget={max_approx_budget}, iterations={iterations}, "
        f"small_graph={small_graph}, index={index}, max_order={max_order}, "
        f"dataset={dataset_name}, model={model_id}, n_layers={n_layers}"
    )
    with tqdm(
        total=total_budget, desc="Running the baseline approximations ", unit=" model calls"
    ) as pbar:
        for approximator, instance_id, budget, iteration, file_name, x_graph in parameter_space:
            run_baseline(
                approximator,
                budget,
                iteration,
                file_name,
                x_graph,
                index=index,
                max_order=max_order,
            )
            pbar.update(budget)


if __name__ == "__main__":

    APPROXIMATORS_TO_RUN = [
        # KernelSHAPIQ.__name__,
        # PermutationSamplingSII.__name__,
        KernelSHAP.__name__,
        PermutationSamplingSV.__name__,
    ]

    approximate_baselines(
        model_id="GCN",  # GCN GAT GIN
        n_layers=2,  # 2 3
        dataset_name="Mutagenicity",  # PROTEINS Mutagenicity
        iterations=2,
        index="SV",
        max_order=1,
        small_graph=False,
        max_approx_budget=10_000,  # 10_000, 2**15
        approximators_to_run=APPROXIMATORS_TO_RUN,
    )
