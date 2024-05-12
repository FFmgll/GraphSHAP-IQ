"""This plotting script takes the interaction values of the approximations and the exact values
from the GraphSHAP-IQ approximations and plots the results."""
import copy
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from shapiq.interaction_values import InteractionValues
from shapiq.moebius_converter import MoebiusConverter
from approximation_utils import create_results_overview_table
from shapiq.games.benchmark.metrics import get_all_metrics


COLORS = {
    "PermutationSamplingSII": "#7d53de",
    "KernelSHAPIQ": "#ff6f00",
    "SVARMIQ": "#00b4d8",
    "GraphSHAPIQ": "#ef27a6",
}

MARKERS = {
    "PermutationSamplingSII": "x",
    "KernelSHAPIQ": "o",
    "SVARMIQ": "d",
    "GraphSHAPIQ": "o",
}


def load_interactions_to_plot(results_df: pd.DataFrame):
    """Load the interaction values of the approximations and the exact values from the GraphSHAP-IQ
    approximations and return the results as a list of dictionaries.

    Args:
        results_df: The results DataFrame.

    Returns:
        The results to plot as a list of dictionaries.
    """
    results_to_plot: list[dict] = []
    # get exact values and transform them with Möbius
    exact_values_index: dict[str, InteractionValues] = {}  # instance_id -> InteractionValues
    instances_exact = results_df[results_df["exact"] == True][["instance_id", "file_path"]]
    for instance_id, file_path in tqdm(
        instances_exact.itertuples(index=False, name=None),
        desc="Transforming exact values",
        total=len(instances_exact),
    ):
        exact = InteractionValues.load(file_path)
        converter = MoebiusConverter(moebius_coefficients=exact)
        exact_values_index[instance_id] = converter(index=INDEX, order=MAX_ORDER)
    print(f"Found {len(exact_values_index)} exact values.")
    for instance_id, values in exact_values_index.items():
        n_players = values.n_players
        print(f"Instance {instance_id} with {n_players} players.")

    # get and transform the graph_shapiq values
    graph_shapiq_values_index: dict[str, InteractionValues] = {}  # instance_id -> InteractionValues
    gshap = results_df[results_df["approximation"] == "GraphSHAPIQ"][
        ["run_id", "instance_id", "budget", "file_path"]
    ]
    for run_id, instance_id, budget, file_path in tqdm(
        gshap.itertuples(index=False, name=None),
        desc="Transforming GraphSHAP-IQ values",
        total=len(gshap),
    ):
        if instance_id not in instances_exact["instance_id"].values:
            print(f"Skipping {instance_id} because exact values are not computed.")
            continue
        graph_shapiq = InteractionValues.load(file_path)
        converter = MoebiusConverter(moebius_coefficients=graph_shapiq)
        graph_shapiq_values_index[instance_id] = converter(index=INDEX, order=MAX_ORDER)

        metrics = get_all_metrics(
            ground_truth=exact_values_index[instance_id],
            estimated=graph_shapiq_values_index[instance_id],
        )
        results_to_plot.append(
            {
                "run_id": run_id,
                "instance_id": instance_id,
                "approximation": "GraphSHAPIQ",
                "budget": budget,
                "index": INDEX,
                "max_order": MAX_ORDER,
                "min_order": 0,
                **metrics,
            }
        )

    # add approximator
    for approx_name in ["PermutationSamplingSII", "KernelSHAPIQ", "SVARMIQ"]:
        approx: dict[str, InteractionValues] = {}  # instance_id - InteractionValues
        kshap = results_df[results_df["approximation"] == approx_name][
            ["run_id", "instance_id", "budget", "file_path"]
        ]
        for run_id, instance_id, budget, file_path in tqdm(
            kshap.itertuples(index=False, name=None),
            desc=f"Loading {approx_name} values",
            total=len(kshap),
        ):
            if instance_id not in instances_exact["instance_id"].values:
                print(f"Skipping {instance_id} because exact values are not computed.")
                continue
            approx[instance_id] = InteractionValues.load(file_path)
            metrics = get_all_metrics(
                ground_truth=exact_values_index[instance_id],
                estimated=approx[instance_id],
            )
            results_to_plot.append(
                {
                    "run_id": run_id,
                    "instance_id": instance_id,
                    "approximation": approx_name,
                    "budget": approx[instance_id].estimation_budget,
                    "index": INDEX,
                    "max_order": MAX_ORDER,
                    "min_order": 0,
                    **metrics,
                }
            )
    return results_to_plot


if __name__ == "__main__":

    APPROX_TO_PLOT = ["PermutationSamplingSII", "SVARMIQ", "KernelSHAPIQ", "GraphSHAPIQ"]

    # plot parameters
    PLOT_METRIC = "SSE"
    CSV_PLOT_FILE = "plot_csv.csv"

    # setting parameters
    MODEL_ID = "GCN"
    DATASET_NAME = "Mutagenicity"
    N_LAYERS = 3
    SMALL_GRAPH = False
    MAX_BUDGET = 2**16

    INDEX = "k-SII"
    MAX_ORDER = 2

    MAX_SIZE = -2

    overview_table = create_results_overview_table()
    overview_table = copy.deepcopy(
        overview_table[
            (overview_table["dataset_name"] == DATASET_NAME)
            & (overview_table["n_layers"] == N_LAYERS)
            & (overview_table["model_id"] == MODEL_ID)
            & (overview_table["small_graph"] == SMALL_GRAPH)
        ]
    )

    # create a DataFrame from the results
    if CSV_PLOT_FILE is None or not os.path.exists(CSV_PLOT_FILE):
        plot_df = pd.DataFrame(load_interactions_to_plot(overview_table))
        plot_df.to_csv("plot_csv.csv", index=False)
    else:
        plot_df = pd.read_csv(CSV_PLOT_FILE)

    # inner join with the overview table and drop all duplicate columns from the overview table
    merge_columns = ["run_id"]
    plot_df = pd.merge(plot_df, overview_table, on=merge_columns, how="inner")
    for column in plot_df.columns:
        if column.endswith("_y"):
            plot_df = plot_df.drop(column, axis=1)
        if column.endswith("_x"):
            plot_df = plot_df.rename(columns={column: column[:-2]})

    # for graphshapiq, only select the values max_neighborhood_size - 1 for each instance_id
    if MAX_SIZE is not None:
        selection = plot_df[plot_df["approximation"] == "GraphSHAPIQ"]
        for instance_id in selection["instance_id"].unique():
            instance_selection = selection[selection["instance_id"] == instance_id]
            max_max_neighborhood_size = instance_selection["max_neighborhood_size"].max()
            if MAX_SIZE > 0:
                max_size_to_plot = min(MAX_SIZE, max_max_neighborhood_size)
            else:
                max_size_to_plot = max_max_neighborhood_size - abs(MAX_SIZE)
            instance_selection = instance_selection[
                instance_selection["max_neighborhood_size"] == max_size_to_plot
            ]
            plot_df = plot_df.drop(
                plot_df[
                    (plot_df["instance_id"] == instance_id)
                    & (plot_df["approximation"] == "GraphSHAPIQ")
                ].index
            )
            plot_df = pd.concat([plot_df, instance_selection])

    # drop all instances with a budget higher than MAX_BUDGET
    plot_df = plot_df[plot_df["budget"] <= MAX_BUDGET]

    # plot the results as a scatter plot for each approximator for the desired metric
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for approx_method in APPROX_TO_PLOT:
        if approx_method == "exact":
            continue
        approx_df = plot_df[plot_df["approximation"] == approx_method]
        if approx_df.empty:
            continue
        if approx_method != "GraphSHAPIQ":
            # average over the iterations
            approx_df = (
                approx_df.groupby(by=["instance_id", "budget"])
                .agg({PLOT_METRIC: "mean"})
                .reset_index()
            )
            approx_df[PLOT_METRIC] = approx_df[PLOT_METRIC]
        budgets = approx_df["budget"]
        metric = approx_df[PLOT_METRIC]
        ax.scatter(
            budgets,
            metric,
            label=approx_method,
            color=COLORS[approx_method],
            marker=MARKERS[approx_method],
        )

    ax.set_xlabel("Budget")
    ax.set_ylabel(PLOT_METRIC)
    title = (
        f"{INDEX} of order {MAX_ORDER} for {DATASET_NAME} with {MODEL_ID} and {N_LAYERS} layers\n"
        f"(small graph: {SMALL_GRAPH}, neighbors size: {MAX_SIZE})"
    )
    ax.set_title(title)

    # set log scale
    if PLOT_METRIC in ("MSE", "SSE", "MAE"):
        ax.set_yscale("log")
        # ax.set_ylim(1e-1, 1e6)
    if "Precision" in PLOT_METRIC:
        ax.set_ylim(0, 1)
    if "Kendall" in PLOT_METRIC:
        ax.set_ylim(-1, 1)

    ax.set_xlim(0, MAX_BUDGET)

    # place legend lower left
    # plt.legend(loc="lower left")
    plt.show()
