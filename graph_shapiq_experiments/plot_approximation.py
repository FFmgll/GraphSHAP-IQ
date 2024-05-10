"""This plotting script takes the interaction values of the approximations and the exact values
from the GraphSHAP-IQ approximations and plots the results."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from shapiq.interaction_values import InteractionValues
from shapiq.moebius_converter import MoebiusConverter
from utils_approximation import load_all_interaction_values
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


def _get_exact_identifier(approx_identifier: str) -> str:
    """Get the identifier of the exact values from the identifier of the approximations.

    Args:
        approx_identifier: The identifier of the approximation values.

    Returns:
        The identifier of the exact values.
    """
    exact_identifier = approx_identifier.split("_")
    exact_identifier = "_".join(exact_identifier[0:4])
    return exact_identifier


if __name__ == "__main__":

    PLOT_METRIC = "Precision@10"

    MODEL_ID = "GCN"
    DATASET_NAME = "Mutagenicity"
    N_LAYERS = 2

    INDEX = "k-SII"
    MAX_ORDER = 2

    # load the interaction values
    interaction_values: dict[str, dict[str, InteractionValues]] = load_all_interaction_values(
        model_id=MODEL_ID, dataset_name=DATASET_NAME, n_layers=N_LAYERS
    )

    # get exact values and transform them with Möbius
    exact_values_index = {}
    for identifier, exact in tqdm(
        interaction_values["exact"].items(), desc="Transforming exact values"
    ):
        converter = MoebiusConverter(moebius_coefficients=exact)
        exact_values_index[_get_exact_identifier(identifier)] = converter(
            index=INDEX, order=MAX_ORDER
        )

    # get and transform the graph_shapiq values
    graph_shapiq_values_index = {}
    for identifier, graph_shapiq in tqdm(
        interaction_values["GraphSHAPIQ"].items(), desc="Transforming GraphSHAPIQ values"
    ):
        converter = MoebiusConverter(moebius_coefficients=graph_shapiq)
        graph_shapiq_values_index[identifier] = converter(index=INDEX, order=MAX_ORDER)

    # add graph_shapiq values back into the interaction_values
    interaction_values["GraphSHAPIQ"] = graph_shapiq_values_index

    # compute approximation quality
    results = []
    for approx_method in interaction_values.keys():
        if approx_method == "exact":
            continue
        approx_values: dict[str, InteractionValues] = interaction_values[approx_method]
        for identifier, approximation in approx_values.items():
            if approximation.index != INDEX and approximation.max_order != MAX_ORDER:
                print(
                    f"Skipping {approximation.index} with order {approximation.max_order} "
                    f"for {approx_method} and {identifier}"
                )
                continue
            ground_truth = exact_values_index[_get_exact_identifier(identifier)]
            metrics = get_all_metrics(ground_truth=ground_truth, estimated=approximation)
            results.append(
                {
                    "identifier": identifier,
                    "exact_identifier": _get_exact_identifier(identifier),
                    "budget": approximation.estimation_budget,
                    "approximation": approx_method,
                    "index": approximation.index,
                    "max_order": approximation.max_order,
                    "min_order": approximation.min_order,
                    "n_players": approximation.n_players,
                    **metrics,
                }
            )

    # create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # plot the results as a scatter plot for each approximator for the desired metric
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for approx_method in interaction_values.keys():
        if approx_method == "exact":
            continue
        approx_df = results_df[results_df["approximation"] == approx_method]
        if approx_method != "GraphSHAPIQ":
            # average over the iterations
            approx_df = approx_df.groupby(by="budget")[PLOT_METRIC].mean().reset_index()
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

    # set log scale
    if PLOT_METRIC in ("MSE", "SSE", "MAE"):
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1e3)
    if "Precision" in PLOT_METRIC:
        ax.set_ylim(0, 1)
    if "Kendall" in PLOT_METRIC:
        ax.set_ylim(-1, 1)

    plt.legend()
    plt.show()
