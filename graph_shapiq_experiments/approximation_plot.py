"""This plotting script visualizes the approximation qualities as a scatter plot for different
approximation methods and budgets."""

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
    "SVARMIQ": "#00b4d8",
    "SVARM": "#00b4d8",
    "SHAPIQ": "#ef27a6",
    "UnbiasedKernelSHAP": "#ef27a6",
    "GraphSHAPIQ": "#7DCE82",
    "L_Shapley": hex_black,
}

MARKERS = {
    "PermutationSamplingSII": "x",
    "PermutationSamplingSV": "x",
    "KernelSHAPIQ": "o",
    "KernelSHAP": "o",
    "SVARMIQ": "d",
    "SVARM": "d",
    "SHAPIQ": "o",
    "GraphSHAPIQ": "o",
    "L_Shapley": "o",
}


def make_box_plots(plot_df, moebius_plot_df) -> None:
    """Make a plot showing the approximation qualities for different approximation methods as
    box plots in one plot.

    Args:
        plot_df: The DataFrame containing the approximation qualities.
        moebius_plot_df: The DataFrame containing the Moebius approximation qualities.
    """
    # remove outliers for plotting percentile: 0.05 and 0.95
    upper_bound = plot_df[PLOT_METRIC].quantile(0.95)
    lower_bound = plot_df[PLOT_METRIC].quantile(0.05)
    plot_df = plot_df[(plot_df[PLOT_METRIC] <= upper_bound) & (plot_df[PLOT_METRIC] >= lower_bound)]

    # remove the interaction sizes not to plot
    if INTERACTION_SIZE_NOT_TO_PLOT is not None and INTERACTION_SIZE_NOT_TO_PLOT != []:
        plot_df = plot_df[~plot_df["max_interaction_size"].isin(INTERACTION_SIZE_NOT_TO_PLOT)]

    # make a single figure with the box plots and the moebius values (shared x-axis)
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [6, 1]}
    )
    box_axis = axes[0]
    moebius_axis = axes[1]

    # get offsets that center the box plots for each approximation method
    n_approx = len(APPROX_TO_PLOT)
    box_plot_width = 1 / n_approx - 0.05
    approx_position_offsets = []
    for i in range(n_approx):
        approx_position_offsets.append(i * box_plot_width - (n_approx - 1) / (2 * n_approx))

    moebius_axis.axhline(0, color="gray", linewidth=0.5)

    index = 0
    for approx in APPROX_TO_PLOT:
        approx_df = plot_df[plot_df["approximation"] == approx]
        if approx_df.empty:
            continue
        box_axis.boxplot(
            [
                approx_df[approx_df["max_interaction_size"] == size][PLOT_METRIC]
                for size in approx_df["max_interaction_size"].unique()
            ],
            positions=approx_df["max_interaction_size"].unique() + approx_position_offsets[index],
            widths=box_plot_width,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(edgecolor=COLORS[approx], facecolor=COLORS[approx] + "33"),
            whiskerprops=dict(color=COLORS[approx]),
            capprops=dict(color=COLORS[approx]),
            medianprops=dict(color=COLORS[approx]),
            meanprops=dict(marker="o", markerfacecolor=COLORS[approx], markeredgecolor="black"),
        )
        index += 1
        # add empty plot for legend
        box_axis.plot([], [], color=COLORS[approx], label=approx)

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
        budget = np.mean(plot_df[plot_df["max_interaction_size"] == size]["budget"].values)
        x_tick_labels.append(rf"$k$={size}")
    moebius_axis.set_xticklabels(x_tick_labels)
    moebius_axis.set_xlabel("GraphSHAP-IQ Interaction Order")
    moebius_axis.tick_params(axis="x", which="both", bottom=True, top=True)  # xticks above + below

    # add ylabels
    box_axis.set_ylabel(PLOT_METRIC)
    moebius_axis.set_ylabel("Moebius")

    # ad legends (only for the box plots)
    box_axis.legend(loc="best")

    # add title
    box_axis.set_title(TITLE)

    # remove white space between the subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(f"plots/{SAVE_NAME_PREFIX}_box_plots.pdf")
    plt.show()


def make_scatter_plot(plot_df) -> None:
    """Make a scatter plot of the approximation qualities for different approximation methods.

    Args:
        plot_df: The DataFrame containing the approximation qualities.
    """
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
    ax.set_title(TITLE)

    # set log scale
    if PLOT_METRIC in ("MSE", "SSE", "MAE") and LOG_SCALE:
        ax.set_yscale("log")
    if "Precision" in PLOT_METRIC:
        ax.set_ylim(0, 1)
    if "Kendall" in PLOT_METRIC:
        ax.set_ylim(-1, 1)

    ax.set_ylim(Y_LIM)
    ax.set_xlim(0, MAX_BUDGET)

    # place legend
    plt.legend(loc="best")
    plt.savefig(f"plots/{SAVE_NAME_PREFIX}_scatter.pdf")
    plt.show()


def make_errors_at_exact_plot(sv_plot_df, k_sii_plot_df) -> None:
    """Plots the errors of the baseline approximations at the sizes where GraphSHAP-IQ has exact
    values.

    The errors are plotted for each approximation method.

    Args:
        sv_plot_df: The DataFrame containing the approximation of SV estimates.
        k_sii_plot_df: The DataFrame containing the approximation of k-SII estimates.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 7))

    # get a sorted list of the approximations to plot
    approx_to_plot = {
        "SV": [
            "GraphSHAPIQ",
            "KernelSHAP",
            "PermutationSamplingSV",
            "SVARM",
            "UnbiasedKernelSHAP",
            "L_Shapley",
        ],
        "k-SII": [
            "GraphSHAPIQ",
            "KernelSHAPIQ",
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

    widths = 0.6
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
                meanprops=dict(marker="o", markerfacecolor=edge_color, markeredgecolor="black"),
            )
            x_ticks.append(approx_pos)
            x_tick_labels.append(index)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    # plt.xticks(rotation=45)

    # add grid
    ax.yaxis.grid(True)

    ax.set_ylabel(PLOT_METRIC)
    ax.set_title(TITLE)
    plt.tight_layout()
    plt.savefig(f"plots/{SAVE_NAME_PREFIX}_errors_at_exact.pdf")
    plt.show()


if __name__ == "__main__":

    # setting parameters
    MODEL_ID = "GIN"  # GCN GIN GAT
    DATASET_NAME = "PROTEINS"  # Mutagenicity PROTEINS BZR
    N_LAYERS = 2  # 2 3
    SMALL_GRAPH = False  # True False
    INDEX = "k-SII"  # k-SII
    MAX_ORDER = 2  # 2

    # plot parameters
    if INDEX == "SV":
        APPROX_TO_PLOT = [
            "PermutationSamplingSV",
            "KernelSHAP",
            "GraphSHAPIQ",
            "L_Shapley",
        ]
    else:
        APPROX_TO_PLOT = [
            "PermutationSamplingSII",
            "KernelSHAPIQ",
            "GraphSHAPIQ",
        ]

    PLOT_METRIC = "SAE@5"  # MSE, SSE, MAE, Precision@10
    LOAD_FROM_CSV = True  # True False (load the results from a csv file or build it from scratch)
    MIN_ESTIMATES = 2  # n drop all max_interaction_sizes with less than n estimates

    # scatter plot parameters
    SCATTER_PLOT = False  # True False (plot the approximation qualities as a scatter plot)
    MAX_SIZE = None  # None -n to n (select the maximum neighborhood size to plot)
    Y_LIM = None  # None (set the y-axis limits)
    LOG_SCALE = True  # True False (set the y-axis to log scale)
    MAX_BUDGET = 2**15  # 2**15 10_000

    # box plot parameters
    BOX_PLOTS = False  # True False (plot the approximation qualities as box plots)
    # None [n, m] (remove the interaction sizes not to plot)
    INTERACTION_SIZE_NOT_TO_PLOT = None  # [1, 2, 3]

    # errors at exact plot
    PLOT_ERRORS_AT_EXACT = True  # True False (plot the errors at the exact values)

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
        df.groupby(["max_interaction_size", "approximation", "index"])[[PLOT_METRIC, "instance_id"]]
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
        rf"${MAX_ORDER}$-SII"
    TITLE = INDEX_TITLE + f" for {DATASET_NAME} with {N_LAYERS}-layer {MODEL_ID}"

    if SCATTER_PLOT:
        make_scatter_plot(df)

    if BOX_PLOTS:
        make_box_plots(df, moebius_df)

    if PLOT_ERRORS_AT_EXACT:
        make_errors_at_exact_plot(sv_exact_df, k_sii_exact_df)
