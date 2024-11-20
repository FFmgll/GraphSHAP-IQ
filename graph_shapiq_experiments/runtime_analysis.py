import numpy as np
from sympy.combinatorics import Permutation
from tqdm import tqdm
import time

from graph_shapiq_experiments.approximation_run_graphshapiq import run_graph_shapiq_approximations
from shapiq import InteractionValues, KernelSHAPIQ, InconsistentKernelSHAPIQ, PermutationSamplingSII, SHAPIQ
from shapiq.approximator import SVARMIQ
from shapiq.moebius_converter import MoebiusConverter
import matplotlib.pyplot as plt
import os
import torch
from shapiq.explainer.graph import (
    _compute_baseline_value,
    GraphSHAPIQ,
    get_explanation_instances,
    load_graph_model, L_Shapley,
)

import pandas as pd
from graph_shapiq_experiments.approximation_utils import (
    is_game_computed,
    save_interaction_value,
    BudgetError,
    GRAPHSHAPIQ_APPROXIMATION_DIR,
    L_SHAPLEY_APPROXIMATION_DIR,
    pre_select_data_ids,
    OVERVIEW_CSV_FILE,
)
from shapiq.games.benchmark.local_xai import GraphGame


def runtime_analysis(INTERACTION_ORDER):
    results = pd.DataFrame(index=data_ids)
    for game in tqdm(games_to_run, desc="Running the GraphSHAP-IQ approximation"):
        start_graphshapiq = time.time()
        computer = GraphSHAPIQ(game)
        total_budget = computer.total_budget
        moebius, gt_shapley_interactions = computer.explain(
            max_interaction_size=computer.max_size_neighbors,
            order=INTERACTION_ORDER,
            efficiency_routine=EFFICIENCY_MODE
        )
        end_graphshapiq = time.time()
        results.loc[game.game_id,"runtime_graphshapiq"] = end_graphshapiq - start_graphshapiq
        results.loc[game.game_id,"budget"] = total_budget
        results.loc[game.game_id,"n_players"] = game.n_players

        APPROXIMATOR_LIST = [KernelSHAPIQ, SVARMIQ, InconsistentKernelSHAPIQ, SHAPIQ, PermutationSamplingSII]


        for approximator_class in APPROXIMATOR_LIST:
            approximator_name = approximator_class.__name__
            start_approximator = time.time()
            approximator = approximator_class(n=game.n_players, index="k-SII", max_order=INTERACTION_ORDER, moebius_lookup=moebius.interaction_lookup)
            approx_shapley_interactions = approximator.approximate(game=game, budget=total_budget)
            end_approximator = time.time()
            results.loc[game.game_id,"mse_"+approximator_name] = np.mean((approx_shapley_interactions-gt_shapley_interactions).values**2)
            results.loc[game.game_id,"runtime_"+approximator_name] = end_approximator-start_approximator

        #L-Shapley only for SV
        if INTERACTION_ORDER == 1:
            max_size_neighbors = (
                computer.max_size_neighbors
            )  # random number 10 that should always be enough
            approximator_name = L_Shapley.__name__
            for interaction_size in range(1, max_size_neighbors + 10):
                try:
                    start_approximator = time.time()
                    l_shapley = L_Shapley(game, max_budget=total_budget)
                    shapley_values, exceeded_budget = l_shapley.explain(
                        max_interaction_size=interaction_size,
                        break_on_exceeding_budget=True,
                    )
                    end_approximator = time.time()
                    runtime = end_approximator-start_approximator
                except ValueError:
                    break
                budget_used = l_shapley.last_n_model_calls
                shapley_values.estimation_budget = budget_used
                shapley_values.estimated = (
                    False if interaction_size == l_shapley.max_size_neighbors else True
                )
                shapley_values.sparsify(threshold=1e-8)
                if exceeded_budget:
                    break

            results.loc[game.game_id, "mse_" + approximator_name] = np.mean(
                (shapley_values - gt_shapley_interactions).values ** 2)
            results.loc[game.game_id, "runtime_" + approximator_name] = runtime

        results.to_csv(SAVE_PATH + "/" + MODEL_ID + "_" + DATASET_NAME + "_" + str(N_LAYERS) + "_" + str(
            INTERACTION_ORDER) + "_" + str(RUN_ID) + "_" + str(GRAPH_INFORMED) + "_runtime_metrics.csv")

if __name__ == "__main__":

    SAVE_PATH = os.path.join("..","results","runtime_analysis")

    INDEX = "k-SII"
    MODEL_ID = "GCN"  # one of GCN GIN GAT
    EFFICIENCY_MODE = True

    MODEL_TYPE = "GCN"
    DATASET_NAME = "Mutagenicity"
    N_LAYERS = 2
    DATA_ID = 71
    RANDOM_SEED = 10  # random seed for the graph layout
    MAX_BUDGET = 10_000
    # run setup
    N_GAMES = 100
    MAX_N_PLAYERS = 40
    MIN_N_PLAYERS = 20
    SORT_PLAYER = False

    GRAPH_INFORMED = True


    RUN_ID = 0

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

    # load the model and get the prediction --------------------------------------------------------
    model = load_graph_model(
        model_type=MODEL_TYPE,
        dataset_name=DATASET_NAME,
        n_layers=N_LAYERS,
        device=torch.device("cpu" if torch.cuda.is_available() else "cpu"),
    )

    # set the games up for the approximation
    games_to_run = []
    explanation_instances = get_explanation_instances(DATASET_NAME)
    for data_id in data_ids:
        x_graph = explanation_instances[int(data_id)]
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

    print(f"Running the GraphSHAP-IQ approximation on {len(games_to_run)} games.")
    print(f"Game_ids: {[game.game_id for game in games_to_run]}")

    #runtime_analysis(INTERACTION_ORDER=1)
    #runtime_analysis(INTERACTION_ORDER=2)
    runtime_analysis(INTERACTION_ORDER=3)