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
from scipy import stats
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

hex_black = "#000000"

COLORS = {
    "PermutationSamplingSII": "#7d53de",
    "PermutationSamplingSV": "#7d53de",
    "KernelSHAPIQ": "#ff6f00",
    "KernelSHAP": "#ff6f00",
    "InconsistentKernelSHAPIQ": "#ffba08",
    "kADDSHAP": "#ffba08",
    "SVARMIQ": "#00b4d8",
    "SVARM": "#00b4d8",
    "SHAPIQ": "#ef27a6",
    "UnbiasedKernelSHAP": "#ef27a6",
    "GraphSHAPIQ": "#7DCE82",
    "L_Shapley": hex_black,

}


def plot_runtime(INTERACTION_ORDER,x_variable,y_variable):
    results = pd.read_csv(SAVE_PATH + "/" + MODEL_ID + "_" + DATASET_NAME + "_" + str(N_LAYERS) + "_" + str(
        INTERACTION_ORDER) + "_" + str(RUN_ID) + "_runtime_metrics.csv")

    size = 8

    results = results.dropna(subset=[x_variable])
    slope, intercept, r_value, p_value, std_err = stats.linregress(results[x_variable],
                                                                   results["runtime_graphshapiq"])
    marker="o"

    plt.figure()

    if x_variable=="budget" and y_variable=="runtime":
        plt.plot(results[x_variable].unique(), slope * results[x_variable].unique() + intercept, '--', alpha=0.5, lw=1,c=COLORS["GraphSHAPIQ"], label='Linear Fit ($R^2$='+str(np.round(r_value,2))+")")
    if y_variable == "runtime":
        plt.scatter(x=results[x_variable],y=results[y_variable+"_graphshapiq"],marker=marker,label="GraphSHAP-IQ",c=COLORS["GraphSHAPIQ"],s=size)
    plt.scatter(x=results[x_variable],y=results[y_variable+"_KernelSHAPIQ"],marker=marker,label="KernelSHAP-IQ",c=COLORS["KernelSHAPIQ"],s=size)
    plt.scatter(x=results[x_variable],y=results[y_variable+"_SVARMIQ"],marker=marker,label="SVARM-IQ",s=size,c=COLORS["SVARMIQ"])
    plt.scatter(x=results[x_variable],y=results[y_variable+"_SHAPIQ"],marker=marker,label="SHAP-IQ",s=size,c=COLORS["SHAPIQ"])
    plt.scatter(x=results[x_variable],y=results[y_variable+"_PermutationSamplingSII"],marker=marker,label="Permutation Sampling",s=size,c=COLORS["PermutationSamplingSII"])
    plt.scatter(x=results[x_variable],y=results[y_variable+"_InconsistentKernelSHAPIQ"],marker=marker,label="Inc. KernelSHAP-IQ",s=size,c=COLORS["InconsistentKernelSHAPIQ"])
    if INTERACTION_ORDER == 1:
        plt.scatter(x=results[x_variable], y=results[y_variable + "_L_Shapley"], marker=marker,
                    label="L-Shapley", c=COLORS["L_Shapley"], s=size)

    if y_variable == "runtime":
        plt.title("Runtime Analysis for Explanation Order " + str(INTERACTION_ORDER))
        plt.ylim(0,5)
        plt.ylabel("Runtime (in Seconds)")

    if y_variable == "mse":
        plt.title("Mean-Squared Error (MSE) for Explanation Order " + str(INTERACTION_ORDER))
        plt.ylim(0,1)
        plt.ylabel("MSE")

    if x_variable == "n_players":
        plt.xlabel("Number of Graph Nodes (n)")
        plt.xticks(np.arange(results[x_variable].min(),results[x_variable].max()+1,1))
    else:
        plt.xlabel("Number of Model Calls")

    plt.legend()
    plt.savefig(SAVE_PATH + "/" + MODEL_ID + "_" + DATASET_NAME + "_" + str(N_LAYERS) + "_" + str(
            INTERACTION_ORDER) + "_" + str(RUN_ID) + "_"+y_variable+"_plot_"+x_variable+".png")
    plt.show()

if __name__=="__main__":
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

    MAX_INTERACTION_ORDER = 3

    RUN_ID = 0

    for INTERACTION_ORDER in range(1,MAX_INTERACTION_ORDER+1):
        plot_runtime(INTERACTION_ORDER,"budget","runtime")
        plot_runtime(INTERACTION_ORDER,"n_players","runtime")
        plot_runtime(INTERACTION_ORDER,"budget","mse")
        plot_runtime(INTERACTION_ORDER,"n_players","mse")
