"""This plotting script visualizes the approximation qualities as a scatter plot for different
approximation methods and budgets."""

from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from approximation_utils_plot import get_plot_df

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

MARKERS = {
    "PermutationSamplingSII": "X",
    "PermutationSamplingSV": "X",
    "KernelSHAPIQ": "v",
    "KernelSHAP": "v",
    "InconsistentKernelSHAPIQ": "v",
    "kADDSHAP": "v",
    "SVARMIQ": "d",
    "SVARM": "d",
    "UnbiasedKernelSHAP": "P",
    "SHAPIQ": "P",
    "GraphSHAPIQ": "o",
    "L_Shapley": "o",
}

# a dict denoting marker sizes for each approximation method
MARKER_SIZES, DEFAULT_MARKER_SIZES = {"GraphSHAPIQ": 8}, 8
LINE_WIDTHS, DEFAULT_LINE_WIDTH = {"GraphSHAPIQ": 3}, 3
METHOD_NAME_MAPPING = {
    "PermutationSamplingSII": "Permutation Sampling",
    "PermutationSamplingSV": "Permutation Sampling",
    "KernelSHAPIQ": "KernelSHAP-IQ",
    "KernelSHAP": "KernelSHAP",
    "InconsistentKernelSHAPIQ": "Inc. KernelSHAP-IQ",
    "kADDSHAP": "k-add. SHAP",
    "SVARMIQ": "SVARM-IQ",
    "SVARM": "SVARM",
    "UnbiasedKernelSHAP": "Unbiased KernelSHAP",
    "SHAPIQ": "SHAP-IQ",
    "GraphSHAPIQ": "GraphSHAP-IQ",
    "L_Shapley": "L-Shapley",
}


def make_box_plots(plot_df, moebius_plot_df) -> None:
    """Make a plot showing the approximation qualities for different approximation methods as
    box plots in one plot.

    Args:
        plot_df: The DataFrame containing the approximation qualities.
        moebius_plot_df: The DataFrame containing the Moebius approximation qualities.
    """
    approx_to_plot = APPROX_TO_PLOT_K_SII if INDEX == "k-SII" else APPROX_TO_PLOT_SV

    # remove outliers for plotting percentile: 0.05 and 0.95
    upper_bound = plot_df[PLOT_METRIC].quantile(0.95)
    lower_bound = plot_df[PLOT_METRIC].quantile(0.05)
    plot_df = plot_df[
        (plot_df[PLOT_METRIC] <= upper_bound) & (plot_df[PLOT_METRIC] >= lower_bound)
    ]

    # remove the interaction sizes not to plot
    if INTERACTION_SIZE_NOT_TO_PLOT is not None and INTERACTION_SIZE_NOT_TO_PLOT != []:
        plot_df = plot_df[
            ~plot_df["max_interaction_size"].isin(INTERACTION_SIZE_NOT_TO_PLOT)
        ]

    # make a single figure with the box plots and the moebius values (shared x-axis)
    fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [6, 1]})
    box_axis = axes[0]
    moebius_axis = axes[1]

    # get offsets that center the box plots for each approximation method
    n_approx = len(approx_to_plot)
    box_plot_width = 1 / n_approx - 0.05
    approx_position_offsets = []
    for i in range(n_approx):
        approx_position_offsets.append(
            i * box_plot_width - (n_approx - 1) / (2 * n_approx)
        )

    moebius_axis.axhline(0, color="gray", linewidth=0.5)

    index = 0
    for approx in approx_to_plot:
        approx_df = plot_df[plot_df["approximation"] == approx]
        if approx_df.empty:
            continue
        box_axis.boxplot(
            [
                approx_df[approx_df["max_interaction_size"] == size][PLOT_METRIC]
                for size in approx_df["max_interaction_size"].unique()
            ],
            positions=approx_df["max_interaction_size"].unique()
            + approx_position_offsets[index],
            widths=box_plot_width,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(edgecolor=COLORS[approx], facecolor=COLORS[approx] + "33"),
            whiskerprops=dict(color=COLORS[approx]),
            capprops=dict(color=COLORS[approx]),
            medianprops=dict(color=COLORS[approx]),
            meanprops=dict(
                marker="o", markerfacecolor=COLORS[approx], markeredgecolor="black"
            ),
        )
        index += 1
        # add empty plot for legend
        if approx == "GraphSHAPIQ":
            box_axis.plot(
                [],
                [],
                color=COLORS[approx],
                label=approx,
                marker=MARKERS[approx],
                linewidth=LINE_WIDTHS.get(approx, DEFAULT_LINE_WIDTH),
                markersize=MARKER_SIZES.get(approx, DEFAULT_MARKER_SIZES),
                mec="white",
            )

    max_size = max(plot_df["max_interaction_size"].unique())
    min_size = min(plot_df["max_interaction_size"].unique())

    # plot the moebius values also as box plots
    moebius_plot_df = moebius_plot_df[
        (moebius_plot_df["size"] >= min_size) & (moebius_plot_df["size"] <= max_size)
    ]
    # moebius_plot_df["value"] = moebius_plot_df["value"].abs()
    moebius_axis.boxplot(
        [
            moebius_plot_df[moebius_plot_df["size"] == size]["value"]
            for size in moebius_plot_df["size"].unique()
        ],
        positions=moebius_plot_df["size"].unique(),
        widths=box_plot_width,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(edgecolor="black", facecolor=hex_black + "33"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
        meanprops=dict(marker="o", markerfacecolor="black", markeredgecolor="black"),
    )

    # add grid
    box_axis.yaxis.grid(True)
    box_axis.set_axisbelow(True)

    # set the x ticks to the max_interaction_size
    moebius_axis.set_xticks(range(min_size, max_size + 1))
    x_tick_labels = []
    for size in range(min_size, max_size + 1):
        budget = np.mean(
            plot_df[plot_df["max_interaction_size"] == size]["budget"].values
        )
        x_tick_labels.append(rf"$\lambda$={size}")
    moebius_axis.set_xticklabels(x_tick_labels)
    moebius_axis.set_xlabel("GraphSHAP-IQ Interaction Order")
    moebius_axis.tick_params(
        axis="x", which="both", bottom=True, top=True
    )  # xticks above + below

    # add ylabels
    box_axis.set_ylabel(PLOT_METRIC)
    moebius_axis.set_ylabel("Moebius")

    # ad legends (only for the box plots)
    box_axis.legend(loc="best")

    # add title
    box_axis.set_title(INDEX_TITLE + " for " + TITLE)

    # set moebius to normal notation (not scientific)
    moebius_axis.ticklabel_format(axis="y", style="plain")
    # replace the y-ticklabels with 10^x
    y_ticks = []
    for tick in moebius_axis.get_yticks():
        if tick == 0:
            y_ticks.append("0")
        if tick > 0:
            y_ticks.append(f"$10^{{{int(np.log10(tick))}}}$")
        if tick < 0:
            y_ticks.append(f"$-10^{{{int(np.log10(-tick))}}}$")
    moebius_axis.set_yticklabels(y_ticks)

    # make log scale
    box_axis.set_yscale("log")

    # remove white space between the subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if SAVE_FIG:
        plt.savefig(f"plots/{SAVE_NAME_PREFIX}_box_plots.pdf")
    plt.show()


def make_scatter_plot(
    plot_df,
    baseline_scatter: bool = True,
    size_alpha: bool = False,
    exact_budget: Optional[int] = None,
) -> None:
    """Make a scatter plot of the approximation qualities for different approximation methods.

    Args:
        plot_df: The DataFrame containing the approximation qualities.
        baseline_scatter: Whether to plot the baseline approximations as scatter points. Defaults to
            `True`.
        size_alpha: Whether to set the alpha value of the scatter points dependent on the size.
            Defaults to `False`.
        exact_budget: The mean budget where GraphSHAP-IQ has exact values. If provided, a vertical
            line is plotted at this budget. Defaults to `None`.
    """
    # drop all instances with a budget higher than MAX_BUDGET
    plot_df = plot_df[plot_df["budget"] <= MAX_BUDGET]

    # remove the interaction sizes not to plot
    if MIN_SIZE_TO_PLOT_SCATTER is not None:
        plot_df = plot_df[plot_df["max_interaction_size"] >= MIN_SIZE_TO_PLOT_SCATTER]

    # used if the alpha value of the scatter points should be dependent on the size
    max_interaction_size = int(np.max(plot_df["max_interaction_size"]))
    alphas_per_size = {
        int(size): float(size / max_interaction_size)
        for size in plot_df["max_interaction_size"].unique()
    }

    # plot the results as a scatter plot for each approximator for the desired metric
    fig, ax = plt.subplots(1, 1)
    approx_to_plot = APPROX_TO_PLOT_K_SII if INDEX == "k-SII" else APPROX_TO_PLOT_SV
    for approx_method in approx_to_plot:
        if approx_method == "exact":
            continue
        approx_df = plot_df[plot_df["approximation"] == approx_method]
        if approx_df.empty:
            continue
        budgets_mean, budgets_sem, metrics_mean, metrics_sem = [], [], [], []
        for size in approx_df["max_interaction_size"].unique():
            size_approx_df = approx_df[approx_df["max_interaction_size"] == size]
            alpha = alphas_per_size[size] if size_alpha else 0.25
            budgets = size_approx_df["budget"]
            metric = size_approx_df[PLOT_METRIC]
            budgets_mean.append(budgets.mean()), budgets_sem.append(budgets.sem())
            metrics_mean.append(metric.mean()), metrics_sem.append(metric.sem())
            if not baseline_scatter and approx_method != "GraphSHAPIQ":
                continue
            ax.plot(
                budgets,
                metric,
                color=COLORS[approx_method],
                marker=MARKERS[approx_method],
                alpha=alpha,
                linewidth=0,
                markersize=MARKER_SIZES.get(approx_method, DEFAULT_MARKER_SIZES),
            )

        # plot the line plots and error bars with whiskers
        ax.errorbar(
            budgets_mean,
            metrics_mean,
            xerr=budgets_sem,
            yerr=metrics_sem,
            color=COLORS[approx_method],
            marker=MARKERS[approx_method],
            linestyle="solid",
            linewidth=LINE_WIDTHS.get(approx_method, DEFAULT_LINE_WIDTH),
            mec="white",
            markersize=MARKER_SIZES.get(approx_method, DEFAULT_MARKER_SIZES) + 2,
        )
        ax.fill_between(
            budgets_mean,
            np.array(metrics_mean) - np.array(metrics_sem),
            np.array(metrics_mean) + np.array(metrics_sem),
            alpha=0.25,
            color=COLORS[approx_method],
        )

    if exact_budget is not None:
        ax.axvline(
            exact_budget, color=hex_black + "33", linestyle="solid", linewidth=0.75
        )
        # add a text label for the exact budget and adjust it to the left
        ax.text(
            exact_budget,
            Y_LIM[1] * 0.05,
            f"GraphSHAP-IQ becomes exact\n after {exact_budget} evaluations",
            color=hex_black + "55",
            fontsize=11,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.2"),
        )

    # manually add GraphSHAP-IQ to the legend
    _add_approx_to_legend(ax, "GraphSHAPIQ")

    ax.set_xlabel("Model Evaluations")
    ax.set_ylabel(PLOT_METRIC)
    ax.set_title(INDEX_TITLE + " for " + TITLE)

    # remove the upper and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # set log scale
    if PLOT_METRIC in ("MSE", "SSE", "MAE") and LOG_SCALE:
        ax.set_yscale("log")
    if "Precision" in PLOT_METRIC:
        ax.set_ylim(0, 1)
    if "Kendall" in PLOT_METRIC:
        ax.set_ylim(-1, 1)

    # set y-axis limits
    ax.set_ylim(Y_LIM)

    # place legend lower left
    ax.legend(loc="lower left")

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(f"plots/{SAVE_NAME_PREFIX}_scatter.pdf")
    plt.show()


def make_errors_at_exact_plot(
    sv_plot_df, k_sii_plot_df, l_shapley_df, log_scale: bool = False
) -> None:
    """Plots the errors of the baseline approximations at the sizes where GraphSHAP-IQ has exact
    values.

    The errors are plotted for each approximation method.

    Args:
        sv_plot_df: The DataFrame containing the approximation of SV estimates.
        k_sii_plot_df: The DataFrame containing the approximation of k-SII estimates.
        l_shapley_df: The DataFrame containing the approximation of L-Shapley estimates.
        log_scale: Whether to set the y-axis to log scale. Defaults to `False`.
    """
    fig, ax = plt.subplots(1, 1)

    # get a sorted list of the approximations to plot
    approx_to_plot = {
        "SV": [
            "GraphSHAPIQ",
            "KernelSHAP",
            "kADDSHAP",
            "PermutationSamplingSV",
            "SVARM",
            "UnbiasedKernelSHAP",
            "L_Shapley",
        ],
        "k-SII": [
            "GraphSHAPIQ",
            "KernelSHAPIQ",
            "InconsistentKernelSHAPIQ",
            "PermutationSamplingSII",
            "SVARMIQ",
            "SHAPIQ",
        ],
    }
    exact_dfs = {"SV": sv_plot_df, "k-SII": k_sii_plot_df}
    n_approx_max = max(len(approx_to_plot["SV"]), len(approx_to_plot["k-SII"]))

    # print the avg number of budgets for SV and k-SII
    print("SV avg number of budgets:", sv_plot_df["budget"].mean())
    print("k-SII avg number of budgets:", k_sii_plot_df["budget"].mean())

    widths = 0.75
    x_ticks, x_tick_labels = [], []
    for position in range(n_approx_max):
        for index, df_exact in exact_dfs.items():
            try:
                approx_method = approx_to_plot[index][position]
            except IndexError:  # not enough approximations in this index
                continue
            approx_df = df_exact[df_exact["approximation"] == approx_method]
            if approx_df.empty:
                continue
            if index == "SV":
                approx_pos = position * 1.5 - widths / 2
                errors = approx_df[PLOT_METRIC]
            else:
                approx_pos = position * 1.5 + widths / 2
                errors = approx_df[PLOT_METRIC]
            if log_scale and approx_method == "GraphSHAPIQ":
                continue
            # plot a boxplot
            color = COLORS[approx_method] + "33"
            edge_color = COLORS[approx_method]
            ax.boxplot(
                errors,
                positions=[approx_pos],
                widths=widths,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(edgecolor=edge_color, facecolor=color),
                whiskerprops=dict(color=edge_color),
                capprops=dict(color=edge_color),
                medianprops=dict(color=edge_color),
                meanprops=dict(
                    marker="o", markerfacecolor=edge_color, markeredgecolor="black"
                ),
            )
            x_ticks.append(approx_pos)
            x_tick_labels.append(index.replace("k-", f"${MAX_ORDER}$-"))

    # manually add l_shapley values where the size is the highest
    l_shapley_errors = []
    for instance_id in l_shapley_df["instance_id"].unique():
        l_shapley_instance = l_shapley_df[l_shapley_df["instance_id"] == instance_id]
        max_size = l_shapley_instance["max_interaction_size"].max()
        error = l_shapley_instance[
            l_shapley_instance["max_interaction_size"] == max_size
        ][PLOT_METRIC]
        if not error.empty:
            l_shapley_errors.append(error.values[0])
    position = (n_approx_max - 1) * 1.5 - widths / 2
    ax.boxplot(
        l_shapley_errors,
        positions=[position],
        widths=widths,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(edgecolor=hex_black, facecolor=hex_black + "33"),
        whiskerprops=dict(color=hex_black),
        capprops=dict(color=hex_black),
        medianprops=dict(color=hex_black),
        meanprops=dict(marker="o", markerfacecolor=hex_black, markeredgecolor="black"),
    )
    x_ticks.append(position)
    x_tick_labels.append("SV")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    # scale the fontsize of the x-ticks down by -2
    ax.tick_params(axis="x", labelsize=plt.rcParams["font.size"] - 2)

    ax.set_xlabel("Approximation Methods")

    # set log scale
    if log_scale:
        ax.set_yscale("log")
    else:
        # add a horizontal line at 0
        ax.axhline(0, color="gray", linewidth=0.5)

    # set ylimits
    ax.set_ylim(Y_LIM_EXACT)

    ax.set_ylabel(PLOT_METRIC)
    ax.set_title(TITLE)
    plt.tight_layout()
    if SAVE_FIG:
        save_name_prefix = SAVE_NAME_PREFIX.replace(f"_{INDEX}_{MAX_ORDER}", f"")
        plt.savefig(f"plots/{save_name_prefix}_errors_at_exact.pdf")
    plt.show()


def _add_approx_to_legend(axis, approx_name) -> None:
    axis.plot(
        [],
        [],
        label=METHOD_NAME_MAPPING.get(approx_name, approx_name),
        color=COLORS[approx_name],
        marker=MARKERS[approx_name],
        markersize=MARKER_SIZES.get(approx_name, DEFAULT_MARKER_SIZES) + 1,
        linewidth=LINE_WIDTHS.get(approx_name, DEFAULT_LINE_WIDTH) + 1,
        mec="white",
    )


if __name__ == "__main__":

    # settings for paper plots ---------------------------------------------------------------------

    # 1. scatter plot Mutagenicity
    # MODEL_ID = "GIN", DATASET_NAME = "Mutagenicity", N_LAYERS = 2, SMALL_GRAPH = False,
    # INDEX = "k-SII", MAX_ORDER = 2, PLOT_METRIC = "MSE", MIN_ESTIMATES = 2, Y_LIM = (1e-11, 2e2),
    # LOG_SCALE = True, MIN_SIZE_TO_PLOT_SCATTER = 1, (EXACT_BUDGET = 6836 or None)

    # 2. scatter plot PROTEINS
    # MODEL_ID = "GAT", DATASET_NAME = "PROTEINS", N_LAYERS = 2, SMALL_GRAPH = False,
    # INDEX = "k-SII", MAX_ORDER = 2, PLOT_METRIC = "MSE", MIN_ESTIMATES = 2, Y_LIM = (1e-8, 2e4),
    # LOG_SCALE = True, MIN_SIZE_TO_PLOT_SCATTER = 2, (EXACT_BUDGET = 5090 or None)

    # ----------------------------------------------------------------------------------------------

    # setting parameters  --------------------------------------------------------------------------
    MODEL_ID = "GCN"  # GCN GIN GAT
    DATASET_NAME = "Mutagenicity"  # Mutagenicity PROTEINS BZR
    N_LAYERS = 2  # 2 3
    SMALL_GRAPH = False  # True False
    INDEX = "SV"  # k-SII
    MAX_ORDER = 2  # 2

    # plot parameters  -----------------------------------------------------------------------------
    APPROX_TO_PLOT_SV = [
        "PermutationSamplingSV",
        "KernelSHAP",
        "kADDSHAP",
        "SVARM",
        "UnbiasedKernelSHAP",
        "GraphSHAPIQ",
        "L_Shapley",
    ]
    APPROX_TO_PLOT_K_SII = [
        "PermutationSamplingSII",
        "SHAPIQ",
        "SVARMIQ",
        "InconsistentKernelSHAPIQ",
        "KernelSHAPIQ",
        "GraphSHAPIQ",
    ]

    PLOT_METRIC = "MSE"  # MSE, SSE, MAE, Precision@10
    LOAD_FROM_CSV = (
        True  # True False (load the results from a csv file or build it from scratch)
    )
    MIN_ESTIMATES = 2  # n drop all max_interaction_sizes with less than n estimates
    SAVE_FIG = True  # True False (save the figure as a pdf)
    plt.rcParams.update({"font.size": 16})  # increase the font size of the plot
    plt.rcParams["figure.figsize"] = (8, 7)  # set figure size

    # scatter plot parameters ----------------------------------------------------------------------
    SCATTER_PLOT = (
        True  # True False (plot the approximation qualities as a scatter plot)
    )

    MAX_SIZE = None  # None -n to n (select the maximum neighborhood size to plot)
    Y_LIM = None  # None (set the y-axis limits for the scatter plot)
    LOG_SCALE = True  # True False (set the y-axis to log scale)
    MIN_SIZE_TO_PLOT_SCATTER = 2  # n (minimum size to plot the scatter plot)
    MAX_BUDGET = 10_000
    EXACT_BUDGET = None  # None (set the budget where GraphSHAP-IQ has exact values)

    # box plot parameters
    BOX_PLOTS = True  # True False (plot the approximation qualities as box plots)
    # None [n, m] (remove the interaction sizes not to plot)
    INTERACTION_SIZE_NOT_TO_PLOT = [1, 2]

    # errors at exact plot
    PLOT_ERRORS_AT_EXACT = False  # True False (plot the errors at the exact values)
    Y_LIM_EXACT = (
        -1e-4,
        1e-3,
    )  # None (set the y-axis limits for the errors at the exact values)

    # legend plot parameters
    MAKE_LEGEND_PLOT = False  # True False (make a plot with all the legend elements)

    # sanity check plots
    MAKE_SANITY_CHECK_PLOTS = (
        False  # True False (make a plot with all the legend elements)
    )

    SAVE_NAME_PREFIX = f"{DATASET_NAME}_{MODEL_ID}_{N_LAYERS}_{INDEX}_{MAX_ORDER}"

    k_sii_df, k_sii_moebius_df, k_sii_exact_df = get_plot_df(
        index="k-SII",
        max_order=MAX_ORDER,
        dataset_name=DATASET_NAME,
        n_layers=N_LAYERS,
        model_id=MODEL_ID,
        small_graph=SMALL_GRAPH,
        load_from_csv=LOAD_FROM_CSV,
    )

    sv_df, sv_moebius_df, sv_exact_df = get_plot_df(
        index="SV",
        max_order=1,
        dataset_name=DATASET_NAME,
        n_layers=N_LAYERS,
        model_id=MODEL_ID,
        small_graph=SMALL_GRAPH,
        load_from_csv=LOAD_FROM_CSV,
    )

    df = pd.concat([k_sii_df, sv_df])

    # average the PLOT METRIC over ["instance_id", "budget", "approximation"] but keep all other
    # the df should then be smaller (only average rows)

    aggregation = {PLOT_METRIC: "mean"}
    for column in df.columns:
        if column not in [
            "instance_id",
            "max_interaction_size",
            "approximation",
            "index",
            PLOT_METRIC,
        ]:
            aggregation[column] = "first"
    df = (
        df.groupby(["instance_id", "max_interaction_size", "approximation", "index"])
        .agg(aggregation)
        .reset_index()
    )

    # drop all max_interaction_sizes with less than n estimates
    rows_to_drop = (
        df.groupby(["max_interaction_size", "approximation", "index"])[
            [PLOT_METRIC, "instance_id"]
        ]
        .agg({PLOT_METRIC: "count", "instance_id": "first"})
        .reset_index()
    )
    rows_to_drop = rows_to_drop[rows_to_drop[PLOT_METRIC] < MIN_ESTIMATES]
    for _, row in rows_to_drop.iterrows():
        df = df[
            ~(
                (df["max_interaction_size"] == row["max_interaction_size"])
                & (df["approximation"] == row["approximation"])
                & (df["instance_id"] == row["instance_id"])
                & (df["index"] == row["index"])
            )
        ]

    sv_df = df[df["index"] == "SV"]
    k_sii_df = df[df["index"] == "k-SII"]

    if INDEX == "SV":
        df = sv_df
        moebius_df = sv_moebius_df
    else:
        df = k_sii_df
        moebius_df = k_sii_moebius_df

    # create the titles
    INDEX_TITLE = INDEX
    if INDEX == "k-SII":
        INDEX_TITLE = rf"${MAX_ORDER}$-SII"
    TITLE = f"{DATASET_NAME} with {N_LAYERS}-layer {MODEL_ID}"

    if SCATTER_PLOT:
        make_scatter_plot(df, exact_budget=EXACT_BUDGET)
        # make_scatter_plot(df, baseline_scatter=False)

    if BOX_PLOTS:
        make_box_plots(df, moebius_df)

    if PLOT_ERRORS_AT_EXACT:
        l_shapley = sv_df[sv_df["approximation"] == "L_Shapley"]
        make_errors_at_exact_plot(sv_exact_df, k_sii_exact_df, l_shapley)
        Y_LIM_EXACT = None
        make_errors_at_exact_plot(
            sv_exact_df, k_sii_exact_df, l_shapley, log_scale=True
        )

    if MAKE_SANITY_CHECK_PLOTS:
        # plot for k-SII a simple scatter plot for the budgets as a sanity check
        fig, ax = plt.subplots(1, 1)
        for approx, approx_df in df.groupby("approximation"):
            ax.scatter(
                approx_df["budget"],
                approx_df[PLOT_METRIC],
                label=approx,
                c=COLORS[approx],
            )
        ax.set_xlabel("Budget")
        ax.set_ylabel(PLOT_METRIC)
        ax.set_title(f"k-SII {DATASET_NAME} {MODEL_ID}")
        ax.set_yscale("log")
        ax.legend()
        plt.show()

        # plot for SV a simple scatter plot for the budgets as a sanity check
        fig, ax = plt.subplots(1, 1)
        for approx, approx_df in sv_df.groupby("approximation"):
            ax.scatter(
                approx_df["budget"],
                approx_df[PLOT_METRIC],
                label=approx,
                c=COLORS[approx],
            )
        ax.set_xlabel("Budget")
        ax.set_ylabel(PLOT_METRIC)
        ax.set_title(f"SV {DATASET_NAME} {MODEL_ID}")
        ax.set_yscale("log")
        ax.legend()
        plt.show()

    # make a plot with all the legend elements
    if MAKE_LEGEND_PLOT:
        fig, ax = plt.subplots(1, 1)
        ax.plot([], [], label="$\\bf{Baselines}$", linewidth=0)
        ax.plot([], [], label="$\\bf{SV}$", linewidth=0)
        _add_approx_to_legend(ax, "KernelSHAP")
        _add_approx_to_legend(ax, "kADDSHAP")
        _add_approx_to_legend(ax, "PermutationSamplingSV")
        _add_approx_to_legend(ax, "UnbiasedKernelSHAP")
        _add_approx_to_legend(ax, "SVARM")
        _add_approx_to_legend(ax, "L_Shapley")
        ax.plot([], [], label="$\\bf{2-SII}$", linewidth=0)
        _add_approx_to_legend(ax, "KernelSHAPIQ")
        _add_approx_to_legend(ax, "InconsistentKernelSHAPIQ")
        _add_approx_to_legend(ax, "PermutationSamplingSII")
        _add_approx_to_legend(ax, "SHAPIQ")
        _add_approx_to_legend(ax, "SVARMIQ")
        ax.plot([], [], label="", linewidth=0)
        _add_approx_to_legend(ax, "GraphSHAPIQ")
        ax.legend()
        if SAVE_FIG:
            plt.savefig(f"plots/legend.pdf")
        plt.show()
