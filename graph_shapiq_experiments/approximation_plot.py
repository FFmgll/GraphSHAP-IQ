"""This plotting script visualizes the approximation qualities as a scatter plot for different
approximation methods and budgets."""

import matplotlib.pyplot as plt
import pandas as pd

from approximation_utils_plot import get_plot_df

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


def make_box_plots(plot_df) -> None:
    """Make a plot showing the approximation qualities for different approximation methods as
    box plots in one plot.

    Args:
        plot_df: The DataFrame containing the approximation qualities.
    """
    # remove outliers for plotting percentile: 0.05 and 0.95
    upper_bound = plot_df[PLOT_METRIC].quantile(0.95)
    lower_bound = plot_df[PLOT_METRIC].quantile(0.05)
    plot_df = plot_df[(plot_df[PLOT_METRIC] <= upper_bound) & (plot_df[PLOT_METRIC] >= lower_bound)]

    # remove the interaction sizes not to plot
    if INTERACTION_SIZE_NOT_TO_PLOT is not None and INTERACTION_SIZE_NOT_TO_PLOT != []:
        plot_df = plot_df[~plot_df["max_interaction_size"].isin(INTERACTION_SIZE_NOT_TO_PLOT)]

    # make a single figure with box plots for each apporximation method at each max_interaction_size
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

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
        ax.boxplot(
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
        ax.plot([], [], color=COLORS[approx], label=approx)

    # remove the x ticks
    ax.set_xticks([])
    # set the x ticks to the max_interaction_size
    max_size = max(plot_df["max_interaction_size"].unique())
    min_size = min(plot_df["max_interaction_size"].unique())
    ax.set_xticks(range(min_size, max_size + 1))
    ax.set_xticklabels(range(min_size, max_size + 1))

    # add grid behind the box plots
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    ax.set_xlabel("Interaction size")
    ax.set_ylabel(PLOT_METRIC)
    plt.legend(loc="best")
    small_graph = "small" if SMALL_GRAPH else "large"
    title = (
        f"{INDEX} of order {MAX_ORDER} for {DATASET_NAME} ({small_graph} graph) "
        + f"with {MODEL_ID} ({N_LAYERS} layers)"
    )
    ax.set_title(title)
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
    INDEX = "k-SII"  # k-SII
    MAX_ORDER = 2  # 2

    # plot parameters
    APPROX_TO_PLOT = [
        "PermutationSamplingSII",
        # "SVARMIQ",
        "KernelSHAPIQ",
        "GraphSHAPIQ",
    ]
    PLOT_METRIC = "SSE"  # MSE, SSE, MAE, Precision@10
    LOAD_FROM_CSV = True  # True False (load the results from a csv file or build it from scratch)
    MAX_INTERACTION_SIZES_TO_DROP = 2  # None n (drop the interaction sizes higher than max - n)

    # scatter plot parameters
    SCATTER_PLOT = False  # True False (plot the approximation qualities as a scatter plot)
    MAX_SIZE = None  # None -n to n (select the maximum neighborhood size to plot)
    Y_LIM = None  # None (set the y-axis limits)
    LOG_SCALE = True  # True False (set the y-axis to log scale)
    MAX_BUDGET = 10_000  # 2**15 10_000

    # box plot parameters
    BOX_PLOTS = True  # True False (plot the approximation qualities as box plots)
    INTERACTION_SIZE_NOT_TO_PLOT = [1, 2]  # None [n, m] (remove the interaction sizes not to plot)

    df = get_plot_df(
        index=INDEX,
        max_order=MAX_ORDER,
        dataset_name=DATASET_NAME,
        n_layers=N_LAYERS,
        model_id=MODEL_ID,
        small_graph=SMALL_GRAPH,
        load_from_csv=LOAD_FROM_CSV,
        max_interaction_sizes_to_drop=MAX_INTERACTION_SIZES_TO_DROP,
    )

    if SCATTER_PLOT:
        make_scatter_plot(df)

    if BOX_PLOTS:
        make_box_plots(df)
