import numpy as np
from tqdm import tqdm

from graph_shapiq_experiments.approximation_run_graphshapiq import run_graph_shapiq_approximations
from shapiq import InteractionValues, KernelSHAPIQ
from shapiq.moebius_converter import MoebiusConverter
import matplotlib.pyplot as plt
import os
import torch
from shapiq.explainer.graph import (
    _compute_baseline_value,
    GraphSHAPIQ,
    get_explanation_instances,
    load_graph_model,
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


def compute_top_k_ratios(INTERACTION_ORDER):
    TOP_K_RATIOS = pd.DataFrame()
    RATIO_OVERLAP_TOTAL = np.zeros(TOP_K_MAX_RANGE)

    for game in tqdm(games_to_run, desc="Running the GraphSHAP-IQ approximation"):
        computer = GraphSHAPIQ(game)
        total_budget = computer.total_budget
        moebius, gt_shapley_interactions = computer.explain(
            max_interaction_size=computer.max_size_neighbors,
            order=INTERACTION_ORDER,
            efficiency_routine=EFFICIENCY_MODE
        )

        approximator = KernelSHAPIQ(n=game.n_players, index="k-SII", max_order=INTERACTION_ORDER)
        approx_shapley_interactions = approximator.approximate(game=game, budget=total_budget)

        for k in range(1, TOP_K_MAX_RANGE + 1):
            ground_truth_top_k = gt_shapley_interactions.get_top_k_interactions(k)
            approx_top_k = approx_shapley_interactions.get_top_k_interactions(k)
            n_overlap_interactions = len(
                set(ground_truth_top_k.interaction_lookup.keys()).intersection(
                    set(approx_top_k.interaction_lookup.keys())
                )
            )
            RATIO_OVERLAP_TOTAL[k - 1] = n_overlap_interactions / k
        TOP_K_RATIOS[game.game_id] = RATIO_OVERLAP_TOTAL
        TOP_K_RATIOS.to_csv(SAVE_PATH + "/" + MODEL_ID + "_" + DATASET_NAME + "_" + str(N_LAYERS) + "_" + str(
            INTERACTION_ORDER) + "_TOPK_RATIOS.csv")

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

    TOP_K_MAX_RANGE = 30

    #compute_top_k_ratios(INTERACTION_ORDER=1)
    #compute_top_k_ratios(INTERACTION_ORDER=2)
    compute_top_k_ratios(INTERACTION_ORDER=3)




    #SPECIFIC INSTANCE FROM INTRO PLOT

    PLOT_DIR = os.path.join("..", "results", "explanation_graphs")
    ground_truth_filename = "GCN_Mutagenicity_2_71.interaction_values"
    approximation_filename = "GCN_Mutagenicity_2_71_approximation.interaction_values"
    ground_truth_mi = InteractionValues.load_interaction_values(
        os.path.join(PLOT_DIR, ground_truth_filename)
    )
    approx_3sii = InteractionValues.load_interaction_values(
        os.path.join(PLOT_DIR, approximation_filename)
    )

    converter = MoebiusConverter(ground_truth_mi)
    ground_truth_3sii = converter.moebius_to_shapley_interaction(index="k-SII", order=3)

    TOP_K_MAX_RANGE = 30
    TOP_K_RATIOS = pd.DataFrame()
    INTERACTION_ORDER = 3
    DATA_ID = "71"
    RATIO_OVERLAP_TOTAL = np.zeros(TOP_K_MAX_RANGE)
    for k in range(1, TOP_K_MAX_RANGE + 1):
        ground_truth_top_k = ground_truth_3sii.get_top_k_interactions(k)
        approx_top_k = approx_3sii.get_top_k_interactions(k)
        n_overlap_interactions = len(
            set(ground_truth_top_k.interaction_lookup.keys()).intersection(
                set(approx_top_k.interaction_lookup.keys())
            )
        )
        RATIO_OVERLAP_TOTAL[k - 1] = n_overlap_interactions / k

    TOP_K_RATIOS[DATA_ID] = RATIO_OVERLAP_TOTAL
    TOP_K_RATIOS.to_csv(SAVE_PATH + "/GCN_Mutagenicity_2_"+DATA_ID+"_TOPK_RATIOS.csv")

