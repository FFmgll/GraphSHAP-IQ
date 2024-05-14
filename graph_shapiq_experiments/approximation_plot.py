"""This plotting script visualizes the approximation qualities as a scatter plot for different
approximation methods and budgets."""

import matplotlib.pyplot as plt
import pandas as pd

from approximation_utils_plot import get_plot_df

COLORS = {
    "PermutationSamplingSII": "#7d53de",
    "PermutationSamplingSV": "#7d53de",
    "KernelSHAPIQ": "#ff6f00",
    "KernelSHAP": "#ff6f00",
    "SVARMIQ": "#00b4d8",
    "SVARM": "#00b4d8",
    "GraphSHAPIQ": "#ef27a6",
}

hex_black = "#000000"

MARKERS = {
    "PermutationSamplingSII": "x",
    "PermutationSamplingSV": "x",
    "KernelSHAPIQ": "o",
    "KernelSHAP": "o",
    "SVARMIQ": "d",
    "SVARM": "d",
    "GraphSHAPIQ": "o",
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
    # select the correct sizes
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

    # moebius_axis.b(
    #    moebius_plot_df["size"],
    #    moebius_plot_df["value"],
    #    label="Moebius",
    #    color="black",
    #    marker="o",
    # )

    # add grid behind the box plots
    box_axis.yaxis.grid(True)
    box_axis.set_axisbelow(True)

    # set the x ticks to the max_interaction_size

    moebius_axis.set_xticks(range(min_size, max_size + 1))
    moebius_axis.set_xticklabels(range(min_size, max_size + 1))
    moebius_axis.set_xlabel("Interaction size")
    moebius_axis.tick_params(axis="x", which="both", bottom=True, top=True)  # xticks above + below

    # add ylabels
    box_axis.set_ylabel(PLOT_METRIC)
    moebius_axis.set_ylabel("Moebius")

    # ad legends (only for the box plots)
    box_axis.legend(loc="best")

    # add title
    small_graph = "small" if SMALL_GRAPH else "large"
    title = (
        f"{INDEX} of order {MAX_ORDER} for {DATASET_NAME} ({small_graph} graph) "
        + f"with {MODEL_ID} ({N_LAYERS} layers)"
    )
    box_axis.set_title(title)

    # remove white space between the subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()


def make_scatter_plot(plot_df) -> None:
    """Make a scatter plot of the approximation qualities for different approximation methods.

    Args:
        plot_df: The DataFrame containing the approximation qualities.
    """
    # for graphshapiq, only select the values max_neighborhood_size - 1 for each instance_id
    if MAX_SIZE is not None:
        selection = plot_df[plot_df["approximation"] == "GraphSHAPIQ"]
        for instance_id in selection["instance_id"].unique():
            instance_selection = selection[selection["instance_id"] == instance_id]
            max_max_neighborhood_size = instance_selection["max_neighborhood_size"].max()
            if MAX_SIZE == 0:
                max_size_to_plot = max_max_neighborhood_size
            elif MAX_SIZE > 0:
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
    plt.show()


if __name__ == "__main__":

    # setting parameters
    MODEL_ID = "GAT"  # GCN GIN GAT
    DATASET_NAME = "PROTEINS"  # Mutagenicity PROTEINS
    N_LAYERS = 2  # 2 3
    SMALL_GRAPH = False  # True False
    INDEX = "SV"  # k-SII
    MAX_ORDER = 1  # 2

    # plot parameters
    if INDEX == "SV":
        APPROX_TO_PLOT = [
            "PermutationSamplingSV",
            "KernelSHAP",
            "GraphSHAPIQ",
        ]
    else:
        APPROX_TO_PLOT = [
            "PermutationSamplingSII",
            "KernelSHAPIQ",
            "GraphSHAPIQ",
        ]

    PLOT_METRIC = "SSE"  # MSE, SSE, MAE, Precision@10
    LOAD_FROM_CSV = True  # True False (load the results from a csv file or build it from scratch)
    MAX_INTERACTION_SIZES_TO_DROP = None  # None n (drop the interaction sizes higher than max - n)

    # scatter plot parameters
    SCATTER_PLOT = True  # True False (plot the approximation qualities as a scatter plot)
    MAX_SIZE = None  # None -n to n (select the maximum neighborhood size to plot)
    Y_LIM = None  # None (set the y-axis limits)
    LOG_SCALE = True  # True False (set the y-axis to log scale)
    MAX_BUDGET = 2**15  # 2**15 10_000

    # box plot parameters
    BOX_PLOTS = True  # True False (plot the approximation qualities as box plots)
    INTERACTION_SIZE_NOT_TO_PLOT = None  # None [n, m] (remove the interaction sizes not to plot)

    df, moebius_df = get_plot_df(
        index=INDEX,
        max_order=MAX_ORDER,
        dataset_name=DATASET_NAME,
        n_layers=N_LAYERS,
        model_id=MODEL_ID,
        small_graph=SMALL_GRAPH,
        load_from_csv=LOAD_FROM_CSV,
        max_interaction_sizes_to_drop=MAX_INTERACTION_SIZES_TO_DROP,
    )

    # average the PLOT METRIC over ["instance_id", "budget", "approximation"] but keep all other
    # the df should then be smaller (only average rows)
    aggregation = {PLOT_METRIC: "mean"}
    for column in df.columns:
        if column not in ["instance_id", "budget", "approximation", PLOT_METRIC]:
            aggregation[column] = "first"
    df = df.groupby(["instance_id", "budget", "approximation"]).agg(aggregation).reset_index()

    if SCATTER_PLOT:
        make_scatter_plot(df)

    if BOX_PLOTS:
        make_box_plots(df, moebius_df)
