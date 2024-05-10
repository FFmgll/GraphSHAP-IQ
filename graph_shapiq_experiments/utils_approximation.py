"""This module contains utility functions for the approximation experiments."""

import os
from typing import Optional

import torch

from shapiq.explainer.graph import (
    get_explanation_instances,
    _compute_baseline_value,
    load_graph_model,
)
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.interaction_values import InteractionValues

BASELINES_DIR = os.path.join("..", "results", "approximation", "baselines")
KERNELSHAPIQ_DIR = os.path.join(BASELINES_DIR, "KernelSHAPIQ")
SVARMIQ_DIR = os.path.join(BASELINES_DIR, "SVARMIQ")
PERMUTATION_DIR = os.path.join(BASELINES_DIR, "PermutationSamplingSII")
EXACT_DIR = os.path.join("..", "results", "approximation", "exact")
GRAPHSHAPIQ_APPROXIMATION_DIR = os.path.join("..", "results", "approximation", "GraphSHAPIQ")

os.makedirs(BASELINES_DIR, exist_ok=True)
os.makedirs(KERNELSHAPIQ_DIR, exist_ok=True)
os.makedirs(SVARMIQ_DIR, exist_ok=True)
os.makedirs(PERMUTATION_DIR, exist_ok=True)
os.makedirs(EXACT_DIR, exist_ok=True)
os.makedirs(GRAPHSHAPIQ_APPROXIMATION_DIR, exist_ok=True)


def parse_file_name(file_name: str) -> dict[str, str]:
    """Parse the file name to get the attributes of the interaction values.

    The naming convention for the file is the following attributes separated by underscore:
        - Type of model, e.g. GCN, GIN
        - Dataset name, e.g. MUTAG
        - Number of Graph Convolutions, e.g. 2 graph conv layers
        - Data ID: a technical identifier of the explained instance
        - Number of players, i.e. number of nodes in the graph
        - Maximum neighbourhood size as integer
        - Efficiency routine used for the approximation as boolean
        - Estimation budget as an integer
        - Iteration number as integer

    Args:
        file_name: The file name of the interaction values.

    Returns:
        The dictionary of attributes.
    """
    if ".interaction_values" in file_name:
        file_name = file_name.replace(".interaction_values", "")
    parts = file_name.split("_")
    return {
        "model_id": str(parts[0]),
        "dataset_name": str(parts[1]),
        "n_layers": int(parts[2]),
        "data_id": int(parts[3]),
        "n_players": int(parts[4]),
        "max_neighborhood_size": int(parts[5]),
        "efficiency": bool(parts[6]),
        "budget": int(parts[7]),
        "iteration": int(parts[8]) if len(parts) == 9 else 1,
    }


def create_file_name(
    model_id: str,
    dataset_name: str,
    n_layers: int,
    data_id: int,
    n_players: int,
    max_neighborhood_size: int,
    efficiency: bool,
    budget: int,
    iteration: int = 1,
) -> str:
    """Create the file name for the interaction values.

    The naming convention for the file is the following attributes separated by underscore:
        - Type of model, e.g. GCN, GIN
        - Dataset name, e.g. MUTAG
        - Number of Graph Convolutions, e.g. 2 graph conv layers
        - Data ID: a technical identifier of the explained instance
        - Number of players, i.e. number of nodes in the graph
        - Maximum neighbourhood size as integer
        - Efficiency routine used for the approximation as boolean
        - Estimation budget as an integer
        - Iteration number as integer

    Args:
        model_id: The model ID.
        dataset_name: The dataset name.
        n_layers: The number of layers.
        data_id: The data ID.
        n_players: The number of players.
        max_neighborhood_size: The maximum neighbourhood size.
        efficiency: Whether the efficiency routine was used for the approximation (True) or not
            (False).
        budget: The estimation budget.
        iteration: The iteration number. Default is 1.

    Returns:
        The file name of the interaction values.
    """
    name = (
        "_".join(
            [
                str(model_id),
                str(dataset_name),
                str(n_layers),
                str(data_id),
                str(n_players),
                str(max_neighborhood_size),
                str(efficiency),
                str(budget),
                str(iteration),
            ]
        )
        + ".interaction_values"
    )
    return name


def save_interaction_value(
    interaction_values: InteractionValues,
    game: GraphGame,
    model_id: str,
    dataset_name: str,
    n_layers: int,
    max_neighborhood_size: int,
    efficiency: bool,
    directory: str = GRAPHSHAPIQ_APPROXIMATION_DIR,
    save_exact: bool = False,
    iteration: int = 1,
) -> None:
    """Save the interaction values to a file.

    Args:
        interaction_values: The interaction values to save.
        game: The game for which the interaction values were computed.
        model_id: The model ID.
        dataset_name: The dataset name.
        n_layers: The number of layers.
        max_neighborhood_size: The maximum neighbourhood size for which the interaction values were
            computed.
        efficiency: Whether the efficiency routine was used for the approximation (True) or not
            (False).
        directory: The directory to save the interaction values in. Default is the
            GRAPHSHAPIQ_APPROXIMATION_DIR.
        save_exact: Whether to save the exact values as well. Default is False. Set to True if you
            evaluate with GraphSHAP-IQ.
        iteration: The iteration number. Default is None.
    """
    save_name = create_file_name(
        model_id,
        dataset_name,
        n_layers,
        game.game_id,
        game.n_players,
        max_neighborhood_size,
        efficiency,
        interaction_values.estimation_budget,
        iteration,
    )
    save_path = os.path.join(directory, save_name)
    interaction_values.save(save_path)
    # check if GT values are available and save them accordingly
    if not interaction_values.estimated and save_exact:
        save_path = os.path.join(EXACT_DIR, save_name)
        interaction_values.save(save_path)


def load_exact_values_from_disk(
    model_id: str, dataset_name: str, n_layers: int
) -> tuple[list[InteractionValues], list[str]]:
    """Load the exact values from disk.

    Args:
        model_id: The model ID.
        dataset_name: The dataset name.
        n_layers: The number of layers.

    Returns:
        The list of exact interaction values and the list of file names.
    """
    interaction_values, file_names = [], []
    for file_name in os.listdir(EXACT_DIR):
        if not file_name.endswith(".interaction_values"):
            continue

        # parse the file name
        parts = file_name.split("_")
        if model_id != parts[0] or dataset_name != parts[1] or n_layers != int(parts[2]):
            continue

        # load the interaction values
        interaction_values.append(InteractionValues.load(os.path.join(EXACT_DIR, file_name)))
        file_names.append(file_name)

    return interaction_values, file_names


def load_approximation_values_from_disk(
    exact_file_name: str, directory: str
) -> list[InteractionValues]:
    """Loads all approximation values from disk that matches the exact file name.

    Args:
        exact_file_name: The file name of the exact values.
        directory: The directory to search for the approximation values in. For example,
            GRAPHSHAPIQ_APPROXIMATION_DIR for the GraphSHAP-IQ approximation values.

    Returns:
        The list of approximation values.
    """
    approximation_values = []
    exact_setup = parse_file_name(exact_file_name)

    for file_name in os.listdir(directory):
        if not file_name.endswith(".interaction_values"):
            continue

        # parse the file name
        file_setup = parse_file_name(file_name)
        if (
            exact_setup["model_id"] == file_setup["model_id"]
            and exact_setup["dataset_name"] == file_setup["dataset_name"]
            and exact_setup["n_layers"] == file_setup["n_layers"]
            and exact_setup["data_id"] == file_setup["data_id"]
        ):
            # load the interaction values
            approximation_values.append(InteractionValues.load(os.path.join(directory, file_name)))

    if len(approximation_values) == 0:
        raise FileNotFoundError(
            f"No approximation values found for {exact_file_name} in {directory}."
        )
    return approximation_values


def is_game_computed(
    model_type: str,
    dataset_name: str,
    n_layers: int,
    data_id: int,
    directory: str = GRAPHSHAPIQ_APPROXIMATION_DIR,
) -> bool:
    """Check whether the game has already been computed."""
    all_files = os.listdir(directory)
    file_part = f"{model_type}_{dataset_name}_{n_layers}_{data_id}"
    for file_name in all_files:
        if file_part in file_name:
            return True
    return False


def is_file_computed(file_name: str, directory: str) -> bool:
    """Check whether the interaction values have already been computed."""
    if directory == BASELINES_DIR:
        # check in all baseline directories
        all_directories = [
            KERNELSHAPIQ_DIR,
            SVARMIQ_DIR,
            PERMUTATION_DIR,
        ]
        for directory in all_directories:
            if is_file_computed(file_name, directory):
                return True
    attributes = parse_file_name(file_name)
    file_part = "_".join(
        [
            attributes["model_id"],
            attributes["dataset_name"],
            str(attributes["n_layers"]),
            str(attributes["data_id"]),
        ]
    )
    all_files = os.listdir(directory)
    for file_name in all_files:
        if file_part in file_name:
            return True
    return False


def get_games_from_file_names(interaction_values_names: list[str]) -> list[GraphGame]:
    """Parses the interaction values names to get the games.

    The naming convention for the file is the following attributes separated by underscore:
        - Type of model, e.g. GCN, GIN
        - Dataset name, e.g. MUTAG
        - Number of Graph Convolutions, e.g. 2 graph conv layers
        - Data ID: a technical identifier of the explained instance
        - Number of players, i.e. number of nodes in the graph
        - Maximum neighbourhood size as integer
        - Efficiency routine used for the approximation as boolean

    Args:
        interaction_values_names: A list of file names of the interaction values.

    Returns:
        The list of games.
    """
    # get the instances and dataset
    game_settings = parse_file_name(interaction_values_names[0])
    dataset_name = game_settings["dataset_name"]
    all_instances = get_explanation_instances(dataset_name=dataset_name)

    # get the games
    games_to_run = []
    for file_name in interaction_values_names:
        if not file_name.endswith(".interaction_values"):
            continue

        # parse the file name
        parts = file_name.split("_")
        model_id = str(parts[0])
        dataset_name = str(parts[1])
        n_layers = int(parts[2])
        data_id = int(parts[3])

        # load the model
        model = load_graph_model(
            model_type=model_id,
            dataset_name=dataset_name,
            n_layers=n_layers,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # set up the game
        x_graph = all_instances[data_id]
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

    return games_to_run


def load_all_interaction_values(
    model_id: str,
    dataset_name: str,
    n_layers: int,
    efficiency: Optional[bool] = None,
) -> dict[str, dict[str, InteractionValues]]:
    """Loads all interaction values from disk that match the given model ID, dataset name number
    of layers and in case of GraphSHAP-IQ the efficiency routine.

    Args:
        model_id: The model ID.
        dataset_name: The dataset name.
        n_layers: The number of layers.
        efficiency: Whether the efficiency routine was used for the approximation (True) or not
            (False). Default is True.

    Returns:
        The dictionary of interaction values mapping from the approximation method to a dictionary
        mapping from the data ID to the list of interaction values.
    """
    interaction_values = {}
    directories = [
        EXACT_DIR,
        KERNELSHAPIQ_DIR,
        SVARMIQ_DIR,
        PERMUTATION_DIR,
        GRAPHSHAPIQ_APPROXIMATION_DIR,
    ]

    for directory in directories:
        approx_method = directory.split(os.sep)[-1]
        interaction_values[approx_method] = {}
        all_files = os.listdir(directory)
        for file_name in all_files:
            if not file_name.endswith(".interaction_values"):
                continue

            # parse the file name
            attributes = parse_file_name(file_name)

            # check if the model ID and dataset name match
            if (
                model_id != attributes["model_id"]
                or dataset_name != attributes["dataset_name"]
                and n_layers != attributes["n_layers"]
            ):
                continue

            # check if the efficiency routine matches
            if (
                directory == GRAPHSHAPIQ_APPROXIMATION_DIR
                and efficiency is not None
                and efficiency != attributes["efficiency"]
            ):
                continue

            file_identifier = "_".join(
                [
                    str(attributes["model_id"]),
                    str(attributes["dataset_name"]),
                    str(attributes["n_layers"]),
                    str(attributes["data_id"]),
                    str(attributes["n_players"]),
                    str(attributes["budget"]),
                    str(attributes["iteration"]),
                ]
            )
            values = InteractionValues.load(os.path.join(directory, file_name))
            interaction_values[approx_method][file_identifier] = values

    return interaction_values


class BudgetError(Exception):
    """Exception raised when the total budget is too high for the approximation."""

    pass
