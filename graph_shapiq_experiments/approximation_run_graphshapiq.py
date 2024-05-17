"""This script runs the GraphSHAP-IQ approximation on different datasets and graphs."""

import copy

import pandas as pd
import torch
from tqdm.auto import tqdm

from approximation_utils import (
    is_game_computed,
    save_interaction_value,
    BudgetError,
    GRAPHSHAPIQ_APPROXIMATION_DIR,
    L_SHAPLEY_APPROXIMATION_DIR,
    pre_select_data_ids,
    OVERVIEW_CSV_FILE,
)
from shapiq.interaction_values import InteractionValues
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.explainer.graph import (
    L_Shapley,
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
        try:
            moebius_values: dict[int, InteractionValues] = run_graph_shapiq_approximation(
                game, efficiency=efficiency
            )
        except BudgetError as e:
            print(e)
            continue
        # save the resulting InteractionValues
        new_file_names = []
        for size, values in moebius_values.items():
            save_name = save_interaction_value(
                interaction_values=values,
                game=game,
                model_id=MODEL_ID,
                dataset_name=DATASET_NAME,
                n_layers=N_LAYERS,
                max_interaction_size=size,
                efficiency=efficiency,
                save_exact=True,
                save_directory=GRAPHSHAPIQ_APPROXIMATION_DIR,
            )
            new_file_names.append(save_name)

        # append the new file names to the csv file if they are not already in there
        if new_file_names:
            # append to OVERVIEW_CSV_FILE
            df = pd.read_csv(OVERVIEW_CSV_FILE)
            for file_name in new_file_names:
                if file_name not in df["file_name"].values:
                    new_df = pd.DataFrame([{"file_name": file_name}])
                    df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(OVERVIEW_CSV_FILE, index=False)


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

    Raises:
        BudgetError: If the total budget is too high.
    """
    approximated_values = {}

    max_order = game.n_players
    approximator = GraphSHAPIQ(game)
    total_budget = approximator.total_budget
    if total_budget > MAX_BUDGET:
        raise BudgetError(
            f"The total budget of {total_budget} is too high for game id {game.game_id}."
        )

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


def run_l_shapley_approximations(games: list[GraphGame]) -> None:
    """Runs the L-Shapley algorithm similarly to the GraphSHAP-IQ approximation.

    Args:
        games: The game to run the approximation on.
    """
    # run the approximation
    for game in tqdm(games, desc="Running the L-Shapley approximation"):
        # compute the approximated values
        approximated_values = {}
        g_shapiq = GraphSHAPIQ(game)
        max_size_neighbors = (
            g_shapiq.max_size_neighbors
        )  # random number 10 that should always be enough
        approximator = L_Shapley(game, max_budget=g_shapiq.total_budget)
        print(
            f"Evaluating Game. max_size_neighbors: {max_size_neighbors}, total_budget: "
            f"{g_shapiq.total_budget}."
        )
        for interaction_size in range(1, max_size_neighbors + 10):
            try:
                shapley_values, exceeded_budget = approximator.explain(
                    max_interaction_size=interaction_size,
                    break_on_exceeding_budget=L_SHAPLEY_BREAK_ON_EXCEEDING_BUDGET,
                )
            except ValueError:
                break
            budget_used = approximator.last_n_model_calls
            shapley_values.estimation_budget = budget_used
            shapley_values.estimated = (
                False if interaction_size == approximator.max_size_neighbors else True
            )
            shapley_values.sparsify(threshold=1e-8)
            approximated_values[interaction_size] = copy.deepcopy(shapley_values)
            print(f"Finished interaction size {interaction_size} with budget {budget_used}")
            if exceeded_budget:
                break
        # save the resulting InteractionValues
        for size, values in approximated_values.items():
            _ = save_interaction_value(
                interaction_values=values,
                game=game,
                model_id=MODEL_ID,
                dataset_name=DATASET_NAME,
                n_layers=N_LAYERS,
                max_interaction_size=size,
                efficiency=False,  # parameter does not exist for L-Shapley
                save_exact=False,  # never save exact values for L-Shapley
                save_directory=L_SHAPLEY_APPROXIMATION_DIR,
            )


if __name__ == "__main__":

    RUN_L_SHAPLEY = True
    L_SHAPLEY_BREAK_ON_EXCEEDING_BUDGET = False  # stops the L-Shapley approximation if necessary

    # run setup
    N_GAMES = 10
    MAX_N_PLAYERS = 40
    MIN_N_PLAYERS = 30

    MODEL_ID = "GCN"  # one of GCN GIN GAT
    DATASET_NAME = "Mutagenicity"  # one of MUTAG PROTEINS ENZYMES AIDS DHFR COX2 BZR Mutagenicity
    SORT_PLAYER = False
    N_LAYERS = 2  # one of 1 2 3
    EFFICIENCY_MODE = True  # one of True False

    max_budget = 2**15
    if DATASET_NAME == "PROTEINS":
        SORT_PLAYER = False
        if N_LAYERS == 1:
            max_budget = 10_000
        elif N_LAYERS == 2:
            max_budget = 10_000
        elif N_LAYERS == 3:
            max_budget = 2**15
        else:
            raise ValueError(f"Wrong Setup for {DATASET_NAME} and {N_LAYERS}")
    if DATASET_NAME == "Mutagenicity":
        SORT_PLAYER = False
        if N_LAYERS == 2:
            max_budget = 10_000
        else:
            raise ValueError(f"Wrong Setup for {DATASET_NAME} and {N_LAYERS}")
    if DATASET_NAME == "BZR":
        SORT_PLAYER = True
        if N_LAYERS == 2:
            max_budget = 2**15
        else:
            raise ValueError(f"Wrong Setup for {DATASET_NAME} and {N_LAYERS}")

    MAX_BUDGET = max_budget

    data_ids = pre_select_data_ids(
        dataset_to_select=DATASET_NAME,
        n_layers=N_LAYERS,
        max_budget=MAX_BUDGET,
        min_players=MIN_N_PLAYERS,
        max_players=MAX_N_PLAYERS,
        sort=SORT_PLAYER,
        sort_budget=False,
    )
    data_ids = data_ids[:N_GAMES]
    print(f"Selected data_ids:", data_ids)

    # see whether a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the model for the approximation
    model = load_graph_model(MODEL_ID, DATASET_NAME, N_LAYERS, device=device)

    # set the games up for the approximation
    games_to_run = []
    explanation_instances = get_explanation_instances(DATASET_NAME)
    for data_id in data_ids:
        x_graph = explanation_instances[int(data_id)]
        if is_game_computed(
            MODEL_ID, DATASET_NAME, N_LAYERS, data_id, directory=GRAPHSHAPIQ_APPROXIMATION_DIR
        ):
            if RUN_L_SHAPLEY:
                pass
            else:
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
        games_to_run.append(game_to_run)
        if len(games_to_run) >= N_GAMES:
            break

    # run the approximation
    if RUN_L_SHAPLEY:
        print(f"Running the L-Shapley approximation on {len(games_to_run)} games.")
        print(f"Game_ids: {[game.game_id for game in games_to_run]}")
        run_l_shapley_approximations(games_to_run)
    else:
        print(f"Running the GraphSHAP-IQ approximation on {len(games_to_run)} games.")
        print(f"Game_ids: {[game.game_id for game in games_to_run]}")
        run_graph_shapiq_approximations(games_to_run, efficiency=EFFICIENCY_MODE)
