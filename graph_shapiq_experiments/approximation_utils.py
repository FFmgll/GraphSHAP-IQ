"""This module contains utility functions for the approximation experiments."""
import glob
import os
from typing import Optional, Union

import pandas as pd
import torch

from shapiq.explainer.graph import _compute_baseline_value, load_graph_model
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.interaction_values import InteractionValues

ALL_SUPPORTED_BASELINE_METHODS = [
    "KernelSHAPIQ",
    "KernelSHAP",
    "SVARMIQ",
    "SVARM",
    "PermutationSamplingSII",
    "PermutationSamplingSV",
    "L_Shapley",
    "InconsistentKernelSHAPIQ",
    "kADDSHAP",
    "UnbiasedKernelSHAP",
    "SHAPIQ",
]

# create directories
this_file_path = os.path.abspath(__file__)
this_file_path = os.path.dirname(this_file_path)
APPROXIMATION_DIR = os.path.join(this_file_path, "..", "results", "approximation")
OVERVIEW_CSV_FILE = os.path.join(APPROXIMATION_DIR, "graph_shapiq_runs.csv")
BASELINES_DIR = os.path.join(APPROXIMATION_DIR, "baselines")
GRAPHSHAPIQ_APPROXIMATION_DIR = os.path.join(APPROXIMATION_DIR, "GraphSHAPIQ")
L_SHAPLEY_APPROXIMATION_DIR = os.path.join(APPROXIMATION_DIR, "baselines", "L_Shapley")
EXACT_DIR = os.path.join(APPROXIMATION_DIR, "exact")
ALL_BASELINE_DIRECTORIES = [
    os.path.join(BASELINES_DIR, method) for method in ALL_SUPPORTED_BASELINE_METHODS
]
ALL_DIRECTORIES = ALL_BASELINE_DIRECTORIES + [
    GRAPHSHAPIQ_APPROXIMATION_DIR,
    EXACT_DIR,
    APPROXIMATION_DIR,
]

for directory in ALL_DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

# create csv if not exist
if not os.path.exists(OVERVIEW_CSV_FILE):
    pd.DataFrame(columns=["file_name"]).to_csv(OVERVIEW_CSV_FILE, index=False)


def parse_file_name(file_name: str) -> dict[str, Union[str, int, bool]]:
    """Parse the file name to get the attributes of the interaction values.

    The naming convention for the file is the following attributes separated by underscore:
        - Type of model, e.g. GCN, GIN
        - Dataset name, e.g. MUTAG
        - Number of Graph Convolutions, e.g. 2 graph conv layers
        - Data ID: a technical identifier of the explained instance
        - Number of players, i.e. number of nodes in the graph
        - Maximum interaction size as integer
        - Efficiency routine used for the approximation as boolean
        - Estimation budget as an integer
        - index of the approximation as string e.g. k-SII or Moebius
        - order of the approximation as integer e.g. 2 or 4
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
        "max_interaction_size": int(parts[5]),
        "efficiency": bool(parts[6]),
        "budget": int(parts[7]),
        "index": str(parts[8]),
        "order": int(parts[9]),
        "iteration": int(parts[10]),
    }


def create_file_name(
    model_id: str,
    dataset_name: str,
    n_layers: int,
    data_id: int,
    n_players: int,
    max_interaction_size: int,
    efficiency: bool,
    budget: int,
    index: str,
    order: int,
    iteration: int,
) -> str:
    """Create the file name for the interaction values.

    The naming convention for the file is the following attributes separated by underscore:
        - Type of model, e.g. GCN, GIN
        - Dataset name, e.g. MUTAG
        - Number of Graph Convolutions, e.g. 2 graph conv layers
        - Data ID: a technical identifier of the explained instance
        - Number of players, i.e. number of nodes in the graph
        - Maximum interaction size as integer
        - Efficiency routine used for the approximation as boolean
        - Estimation budget as an integer
        - index of the approximation as string e.g. k-SII or Moebius
        - order of the approximation as integer e.g. 2 or 4
        - Iteration number as integer

    Args:
        model_id: The model ID.
        dataset_name: The dataset name.
        n_layers: The number of layers.
        data_id: The data ID.
        n_players: The number of players.
        max_interaction_size: The maximum neighbourhood size.
        efficiency: Whether the efficiency routine was used for the approximation (True) or not
            (False).
        budget: The estimation budget.
        index: The index of the approximation.
        order: The order of the approximation.
        iteration: The iteration number.

    Returns:
        The file name of the interaction values.
    """
    if index == "Moebius":
        order = n_players
    name = (
        "_".join(
            [
                str(model_id),
                str(dataset_name),
                str(n_layers),
                str(data_id),
                str(n_players),
                str(max_interaction_size),
                str(efficiency),
                str(budget),
                str(index),
                str(order),
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
    max_interaction_size: int,
    efficiency: bool,
    save_directory: str,
    save_exact: bool = False,
    iteration: int = 1,
    budget: Optional[int] = None,
) -> str:
    """Save the interaction values to a file.

    Args:
        interaction_values: The interaction values to save.
        game: The game for which the interaction values were computed.
        model_id: The model ID.
        dataset_name: The dataset name.
        n_layers: The number of layers.
        max_interaction_size: The maximum neighbourhood size for which the interaction values were
            computed.
        efficiency: Whether the efficiency routine was used for the approximation (True) or not
            (False).
        save_directory: The directory to save the interaction values in. Default is the
            GRAPHSHAPIQ_APPROXIMATION_DIR.
        save_exact: Whether to save the exact values as well. Default is False. Set to True if you
            evaluate with GraphSHAP-IQ.
        iteration: The iteration number. Default is None.
        budget: The estimation budget. Default is None, which uses the budget from the
            interaction_values.

    Returns:
        The file name of the saved interaction values.
    """
    os.makedirs(save_directory, exist_ok=True)
    if budget is None:
        budget = interaction_values.estimation_budget
    save_name = create_file_name(
        model_id,
        dataset_name,
        n_layers,
        game.game_id,
        game.n_players,
        max_interaction_size,
        efficiency,
        budget,
        interaction_values.index,
        interaction_values.max_order,
        iteration,
    )
    save_path = os.path.join(save_directory, save_name)
    interaction_values.save(save_path)
    # check if GT values are available and save them accordingly
    if not interaction_values.estimated and save_exact:
        save_path = os.path.join(EXACT_DIR, save_name)
        interaction_values.save(save_path)
    return save_name


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
    exact_file_name: str, save_directory: str
) -> list[InteractionValues]:
    """Loads all approximation values from disk that matches the exact file name.

    Args:
        exact_file_name: The file name of the exact values.
        save_directory: The directory to search for the approximation values in. For example,
            GRAPHSHAPIQ_APPROXIMATION_DIR for the GraphSHAP-IQ approximation values.

    Returns:
        The list of approximation values.
    """
    approximation_values = []
    exact_setup = parse_file_name(exact_file_name)

    for file_name in os.listdir(save_directory):
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
            approximation_values.append(
                InteractionValues.load(os.path.join(save_directory, file_name))
            )

    if len(approximation_values) == 0:
        raise FileNotFoundError(
            f"No approximation values found for {exact_file_name} in {save_directory}."
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


def is_file_computed(file_name: str, directory: str, min_iterations: int = 1) -> bool:
    """Check whether the interaction values have already been computed at least min_iterations
    times."""
    if directory == BASELINES_DIR:
        # check in all baseline directories
        for directory in ALL_BASELINE_DIRECTORIES:
            if is_file_computed(file_name, directory, min_iterations):
                return True
    attributes = parse_file_name(file_name)
    instance_identifier = "_".join(
        [
            attributes["model_id"],
            attributes["dataset_name"],
            str(attributes["n_layers"]),
            str(attributes["data_id"]),
        ]
    )
    # get all files in the directory
    all_files = os.listdir(directory)
    count_iterations = 0
    for file_name in all_files:
        if instance_identifier in file_name:
            # get iteration number from the file name
            parts = file_name.replace(".interaction_values", "").split("_")
            iteration = int(parts[-1])
            if iteration >= min_iterations:
                count_iterations += 1
    return False


def get_game_from_file_name(interaction_values_name: str, x_graph) -> GraphGame:
    """Parses the interaction values names to get the games.

    Args:
        interaction_values_name: A file name of the interaction values.

    Returns:
        The game instance.
    """
    # get the instances and dataset
    game_settings = parse_file_name(interaction_values_name)
    dataset_name = game_settings["dataset_name"]
    model_id = game_settings["model_id"]
    n_layers = game_settings["n_layers"]
    data_id = game_settings["data_id"]

    # get the instance

    # load the model
    model = load_graph_model(
        model_type=model_id,
        dataset_name=dataset_name,
        n_layers=n_layers,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # set up the game
    baseline = _compute_baseline_value(x_graph)
    game = GraphGame(
        model,
        x_graph=x_graph,
        class_id=x_graph.y.item(),
        max_neighborhood_size=model.n_layers,
        masking_mode="feature-removal",
        normalize=True,
        baseline=baseline,
        instance_id=int(data_id),
    )

    return game


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
    directories = ALL_DIRECTORIES

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


def create_results_overview_table() -> pd.DataFrame:
    """Inspects all approximation directories and creates an overview on what has been computed.

    Returns:
        The DataFrame with the overview.
    """
    results = []
    for directory in ALL_DIRECTORIES:
        is_graphshapiq = directory == GRAPHSHAPIQ_APPROXIMATION_DIR
        approx_method = directory.split(os.sep)[-1]
        all_files = os.listdir(directory)
        for file_name in all_files:
            if not file_name.endswith(".interaction_values"):
                continue

            # parse the file name
            attributes = parse_file_name(file_name)

            # create the instance ID
            instance_id = "_".join(
                [
                    str(attributes["model_id"]),
                    str(attributes["dataset_name"]),
                    str(attributes["n_layers"]),
                    str(attributes["data_id"]),
                ]
            )

            results.append(
                {
                    "run_id": os.path.join(directory, file_name),  # unique identifier
                    "instance_id": instance_id,
                    "model_id": attributes["model_id"],
                    "dataset_name": attributes["dataset_name"],
                    "n_layers": int(attributes["n_layers"]),
                    "data_id": int(attributes["data_id"]),
                    "n_players": int(attributes["n_players"]),
                    "max_interaction_size": int(attributes["max_interaction_size"]),
                    "efficiency": bool(attributes["efficiency"]) if is_graphshapiq else None,
                    "budget": int(attributes["budget"]),
                    "index": attributes["index"],
                    "order": int(attributes["order"]),
                    "iteration": int(attributes["iteration"]),
                    "approximation": approx_method,
                    "small_graph": int(attributes["n_players"]) <= 15,
                    "file_path": os.path.join(directory, file_name),
                    "file_name": file_name,
                    "exact": directory == EXACT_DIR,
                }
            )

    df = pd.DataFrame(results)
    return df


def pre_select_data_ids(
    dataset_to_select: str,
    n_layers: int,
    max_budget: int,
    min_players: int,
    max_players: int,
    sort: bool = False,
    sort_budget: bool = False,
) -> list[int]:
    """Preselect data ids based on the complexity analysis results.

    Args:
        dataset_to_select: The dataset to select the data ids from.
        n_layers: The number of layers to select the data ids from.
        max_budget: The maximum (inclusive) budget to select the data ids from.
        min_players: The minimum (inclusive) number of players to select the data ids from.
        max_players: The maximum (inclusive) number of players to select the data ids from.
        sort: Whether to sort the data ids by the number of players (descending) or not. Default is
            False.
        sort_budget: Whether to sort the data ids by the budget (ascending) or not. Default is
            False.
    """
    # this file path
    this_file_path = os.path.abspath(__file__)
    this_file_path = os.path.dirname(this_file_path)
    save_directory = os.path.join(this_file_path, "..", "results", "complexity_analysis")

    DATASETS = [
        "AIDS",
        "DHFR",
        "COX2",
        "BZR",
        "PROTEINS",
        "ENZYMES",
        "MUTAG",
        "Mutagenicity",
        "AlkaneCarbonyl",
        "Benzene",
        "FluorideCarbonyl",
    ]

    # Import dataset statistics
    dataset_statistics = {}
    for file_path in glob.glob(os.path.join(save_directory, "dataset_statistics", "*.csv")):
        dataset_statistic = pd.read_csv(file_path)
        dataset_name = file_path.split(os.sep)[-1][:-4]
        if dataset_name in DATASETS:
            dataset_statistics[dataset_name] = dataset_statistic

    results = {}
    for file_path in glob.glob(os.path.join(save_directory, "*.csv")):
        result = pd.read_csv(file_path)
        file_name = file_path.split(os.sep)[-1][:-4]  # remove path and ending .csv
        if file_name.split("_")[0] == "complexity" and file_name.split("_")[1] in DATASETS:
            dataset_name = file_name.split("_")[1]
            result["dataset_name"] = dataset_name
            result["n_layers"] = file_name.split("_")[2]
            result = pd.merge(
                result,
                dataset_statistics[dataset_name],
                left_index=True,
                right_index=True,
                how="inner",
            )
            results[file_name] = result
    df = pd.concat(results.values(), keys=results.keys())

    # rename the first column to data_id
    df = df.rename(columns={"Unnamed: 0_x": "data_id"})

    # preselct data ids
    selection = df[df["dataset_name"] == dataset_to_select]
    selection = selection[selection["budget"].astype(float) <= max_budget]
    selection = selection[selection["n_layers"].astype(int) == int(n_layers)]
    selection = selection[
        (max_players >= selection["n_players"].astype(int))
        & (selection["n_players"].astype(int) >= min_players)
    ]

    selection = selection[["data_id", "budget", "n_players"]]
    # sort by n_players descending
    if sort:
        selection = selection.sort_values(by="n_players", ascending=False)

    # sort by budget ascending
    if sort_budget:
        selection = selection.sort_values(by="budget", ascending=True)

    selection.to_csv("selected_data_ids.csv", index=False)

    # return the selected data ids
    return selection["data_id"].tolist()
