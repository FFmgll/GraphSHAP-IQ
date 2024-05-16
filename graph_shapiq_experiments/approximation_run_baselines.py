"""This module runs baseline approximation methods on the same settings as the GraphSHAP-IQ
approximations."""

import copy
import os
import argparse
import sys

import pandas as pd
from tqdm.auto import tqdm

from shapiq.explainer.graph import get_explanation_instances
from shapiq.approximator import (
    KernelSHAPIQ,
    SHAPIQ,
    SVARM,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    PermutationSamplingSII,
    PermutationSamplingSV,
    KernelSHAP,
    kADDSHAP,
    UnbiasedKernelSHAP,
)
from approximation_utils import (
    get_game_from_file_name,
    BASELINES_DIR,
    parse_file_name,
    save_interaction_value,
    EXACT_DIR,
    GRAPHSHAPIQ_APPROXIMATION_DIR,
    APPROXIMATION_DIR,
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
    elif approx_name == SVARM.__name__ and index == "SV" and max_order == 1:
        approximator = SVARM(n=n_players)
    elif approx_name == kADDSHAP.__name__ and index == "SV" and max_order == 1:
        approximator = kADDSHAP(n=n_players, max_order=2)
    elif approx_name == UnbiasedKernelSHAP.__name__ and index == "SV" and max_order == 1:
        approximator = UnbiasedKernelSHAP(n=n_players)
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
    model_id: str,
    dataset_name: str,
    n_layers: int,
    small_graph: bool,
    iterations: list[int],
    approximators_to_run: list[str],
    index: str,
    max_order: int,
    max_approx_budget: int,
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
            for iteration in iterations:
                matched_iteration = [f for f in matched_files if f"_{iteration}." in f]
                if len(matched_iteration) != 0:  # already computed
                    continue
                x_graph = all_instances[attributes["data_id"]]
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
    _print_params(
        model_id,
        dataset_name,
        n_layers,
        small_graph,
        iterations,
        index,
        max_order,
        max_approx_budget,
    )
    with tqdm(
        total=total_budget, desc="Running the baseline approximations ", unit=" model calls"
    ) as pbar:
        for parameters in parameter_space:
            run_baseline(**parameters)
            pbar.update(parameters["budget"])


def _print_params(
    model_id: str,
    dataset_name: str,
    n_layers: int,
    small_graph: bool,
    iterations: list[int],
    index: str,
    max_order: int,
    max_approx_budget: int,
) -> None:
    print(
        f"Settings: max_budget={max_approx_budget}, iterations={iterations}, "
        f"small_graph={small_graph}, index={index}, max_order={max_order}, "
        f"dataset={dataset_name}, model={model_id}, n_layers={n_layers}"
    )


def approximate_parameters(
    model_id: str,
    dataset_name: str,
    n_layers: int,
    small_graph: bool,
    iterations: list[int],
    approximators_to_run: list[str],
    index: str,
    max_order: int,
    max_approx_budget: int,
) -> None:
    """Runs the baseline approximations as specified in the configuration regardless of the already
    computed approximations."""

    parameter_space, total_budget, unique_instances = [], 0, set()
    all_instances = get_explanation_instances(dataset_name=dataset_name)

    # read the csv
    all_graph_shapiq_runs = pd.read_csv(os.path.join(APPROXIMATION_DIR, "graph_shapiq_runs.csv"))
    all_files = set(all_graph_shapiq_runs["file_name"].values)

    for file_name in all_files:
        attributes = parse_file_name(file_name)
        if (
            attributes["model_id"] != model_id
            or attributes["dataset_name"] != dataset_name
            or attributes["n_layers"] != n_layers
            or attributes["budget"] > max_approx_budget
        ):
            continue

        # add to the parameter space
        for approx_name in approximators_to_run:
            for iteration in iterations:
                x_graph = all_instances[attributes["data_id"]]
                params = {
                    "approx_name": approx_name,
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
    _print_params(
        model_id,
        dataset_name,
        n_layers,
        small_graph,
        iterations,
        index,
        max_order,
        max_approx_budget,
    )
    with tqdm(
        total=total_budget, desc="Running the baseline approximations ", unit=" model calls"
    ) as pbar:
        for parameters in parameter_space:
            run_baseline(**parameters)
            pbar.update(parameters["budget"])


if __name__ == "__main__":

    run_from_command_line = False
    print(sys.argv)
    if len(sys.argv) > 2:
        run_from_command_line = True

    # parse the parameters from the command line
    if not run_from_command_line:
        MODEL_ID = "GAT"
        N_LAYERS = 2
        DATASET_NAME = "Mutagenicity"
        ITERATIONS = [1]
        INDEX = "k-SII"
        MAX_ORDER = 2
        SMALL_GRAPH = False
        APPROXIMATE_REGARDLESS = False  # if True, approximate regardless of already approximations
        if INDEX == "k-SII":
            APPROXIMATORS_TO_RUN = [
                # "KernelSHAPIQ",
                # "PermutationSamplingSII",
                # "SVARMIQ",
                "InconsistentKernelSHAPIQ",
                "SHAPIQ",
            ]
        elif INDEX == "SV":
            MAX_ORDER = 1
            APPROXIMATORS_TO_RUN = [
                "KernelSHAP",
                "PermutationSamplingSV",
                "SVARM",
                "UnbiasedKernelSHAP",
                "kADDSHAP",
            ]
        else:
            raise ValueError(f"Index {INDEX} not found. Maybe wrong settings?")

        if not APPROXIMATE_REGARDLESS:
            approximate_baselines(
                model_id=MODEL_ID,
                n_layers=N_LAYERS,  # 2 3
                dataset_name=DATASET_NAME,  # PROTEINS Mutagenicity BZR
                iterations=ITERATIONS,
                index=INDEX,
                max_order=MAX_ORDER,
                small_graph=SMALL_GRAPH,
                max_approx_budget=2**15,
                approximators_to_run=APPROXIMATORS_TO_RUN,
            )
    else:
        # example setting for the command line:
        # k-SII order 2
        # python approximation_run_baselines.py --model_id GCN --dataset_name PROTEINS --approximators_to_use KernelSHAPIQ PermutationSamplingSII SVARMIQ --n_layers 2 --iterations 1 2 --index k-SII --max_order 2
        # SV order 1
        # python approximation_run_baselines.py --model_id GCN --dataset_name PROTEINS --approximators_to_use KernelSHAP PermutationSamplingSV SVARM --n_layers 2 --iterations 1 2 --index SV --max_order 1

        parser = argparse.ArgumentParser()
        parser.add_argument("--model_id", type=str, required=True)
        parser.add_argument("--dataset_name", type=str, required=True)
        parser.add_argument("--approximators_to_use", type=str, nargs="+", required=True)
        parser.add_argument("--n_layers", type=int, required=True)
        parser.add_argument("--iterations", type=int, nargs="+", required=True)
        parser.add_argument("--index", type=str, required=True)
        parser.add_argument("--max_order", type=int, required=True)
        parser.add_argument("--small_graph", type=bool, required=False, default=False)
        args = parser.parse_args()

        MODEL_ID: str = args.model_id  # GCN GAT GIN
        N_LAYERS: int = args.n_layers  # 1 2 3
        DATASET_NAME: str = args.dataset_name  # PROTEINS Mutagenicity BZR
        ITERATIONS: list[int] = args.iterations  # 1 2
        INDEX: str = args.index  # k-SII SV
        MAX_ORDER: int = args.max_order  # 1 2
        SMALL_GRAPH: bool = args.small_graph  # False True
        APPROXIMATORS_TO_RUN: list[str] = args.approximators_to_use

    approximate_parameters(
        model_id=MODEL_ID,
        dataset_name=DATASET_NAME,
        n_layers=N_LAYERS,
        small_graph=SMALL_GRAPH,
        iterations=ITERATIONS,
        approximators_to_run=APPROXIMATORS_TO_RUN,
        index=INDEX,
        max_order=MAX_ORDER,
        max_approx_budget=2**15,
    )
