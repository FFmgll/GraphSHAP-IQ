"""This module runs baseline approximation methods on the same settings as the GraphSHAP-IQ
approximations."""

import copy
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
    EXACT_DIR,
    GRAPHSHAPIQ_APPROXIMATION_DIR,
)


def run_baseline(
    approx_name: str,
    budget: int,
    iteration: int,
    file_name: str,
    x_graph,
    index: str,
    max_order: int,
    interaction_size: int,
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
        budget=interaction_values.estimation_budget,
    )


def approximate_baselines(
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
    # get the dataset
    all_instances = get_explanation_instances(dataset_name=dataset_name)

    # get all files that need to potentially be computed
    file_names = os.listdir(EXACT_DIR) + os.listdir(GRAPHSHAPIQ_APPROXIMATION_DIR)
    file_names = set(file_names)

    # remove all files not matching the model_id, dataset_name, n_layers
    file_names = [
        file_name
        for file_name in file_names
        if f"{model_id}_{dataset_name}_{n_layers}" in file_name
    ]

    # get all files in approximations directory
    approx_files: dict[str, set[str]] = {}
    for approx_name in approximators_to_run:
        save_directory = str(os.path.join(BASELINES_DIR, approx_name))
        approx_files[approx_name] = set(os.listdir(save_directory))

    parameter_space, total_budget, unique_instances = [], 0, set()
    for file_name in file_names:
        attributes = parse_file_name(file_name)
        parts = [
            "model_id",
            "dataset_name",
            "n_layers",
            "data_id",
            "n_players",
            "max_interaction_size",
        ]
        identifier = "_".join(str(attributes[part]) for part in parts)
        for approx_method in approx_files:
            # check if identifier and index and max_order are in the file name
            matched_files = [f for f in approx_files[approx_method] if identifier in f]
            matched_files = [f for f in matched_files if f"{index}_{max_order}" in f]
            if len(matched_files) < iterations:  # needs to be computed
                x_graph = all_instances[attributes["data_id"]]
                for iteration in range(max(len(matched_files), 1), iterations + 1):
                    params = {
                        "approx_name": approx_method,
                        "budget": attributes["budget"],
                        "iteration": iteration,
                        "file_name": file_name,
                        "x_graph": copy.deepcopy(x_graph),
                        "index": index,
                        "max_order": max_order,
                        "interaction_size": attributes["max_interaction_size"],
                    }
                    parameter_space.append(params)
                    total_budget += attributes["budget"]
                unique_instances.add(attributes["data_id"])

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
        for parameters in parameter_space:
            run_baseline(**parameters)
            pbar.update(parameters["budget"])


if __name__ == "__main__":

    APPROXIMATORS_TO_RUN = [
        # KernelSHAPIQ.__name__,
        PermutationSamplingSII.__name__,
        # KernelSHAP.__name__,
        # PermutationSamplingSV.__name__,
    ]

    approximate_baselines(
        model_id="GCN",  # GCN GAT GIN
        n_layers=2,  # 2 3
        dataset_name="PROTEINS",  # PROTEINS Mutagenicity
        iterations=2,
        index="k-SII",
        max_order=2,
        small_graph=False,
        max_approx_budget=10_000,  # 10_000, 2**15
        approximators_to_run=APPROXIMATORS_TO_RUN,
    )
