import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder = "../results/runtime_analysis/"


if __name__ == '__main__':

    add_inset = True
    plt.rcParams.update({"font.size": 20})  # increase the font size of the plot
    plt.rcParams["figure.figsize"] = (8, 7)
    # adjust hatch size
    plt.rcParams["hatch.linewidth"] = 1.5
    MARKER_SIZE = 18
    SCATTER_SIZE = 400

    ORDER_MARKERS = {
        1: "o",
        2: "v",
        3: "X",
    }

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
        "L_Shapley": "black",
    }

    # load the data
    path = os.path.join(data_folder, "GCN_Mutagenicity_2_1_0_runtime_metrics.csv")
    data_first_order = pd.read_csv(path)
    path = os.path.join(data_folder, "GCN_Mutagenicity_2_2_0_runtime_metrics.csv")
    data_second_order = pd.read_csv(path)
    path = os.path.join(data_folder, "GCN_Mutagenicity_2_3_0_runtime_metrics.csv")
    data_third_order = pd.read_csv(path)

    # load informed data
    path = os.path.join(data_folder, "GCN_Mutagenicity_2_2_0_True_runtime_metrics.csv")
    data_second_order_informed = pd.read_csv(path)
    path = os.path.join(data_folder, "GCN_Mutagenicity_2_3_0_True_runtime_metrics.csv")
    data_third_order_informed = pd.read_csv(path)

    metrics = ["mse", "runtime"]

    methods = [
        "InconsistentKernelSHAPIQ",
        "KernelSHAPIQ",
        "SVARMIQ",
        "SHAPIQ",
        "PermutationSamplingSII",
        "L_Shapley",
        #"GraphSHAPIQ",
    ]

    rename_dict = {
        "runtime_graphshapiq": "runtime_GraphSHAPIQ",
        "Unnamed: 0": "instance",
    }

    # rename columns
    data_first_order = data_first_order.rename(columns=rename_dict)
    data_second_order = data_second_order.rename(columns=rename_dict)
    data_third_order = data_third_order.rename(columns=rename_dict)
    data_second_order_informed = data_second_order_informed.rename(columns=rename_dict)
    data_third_order_informed = data_third_order_informed.rename(columns=rename_dict)

    ORDERS = {
        1: data_first_order,
        2: data_second_order,
        3: data_third_order,
    }

    ORDER_NAMES = {
        1: "Order 1 (SV)",
        2: "Order 2 (2-SII)",
        3: "Order 3 (3-SII)",
    }

    fig, axes = plt.subplots(2, 1, sharex=True, height_ratios=[5, 1.2])

    ax = axes[0]
    ax_graphsahpiq = axes[1]
    ax.grid(which="major", linestyle="--", color="gray", lw=0.5)

    for z_ord, method in enumerate(methods, start=1):
        for order in ORDERS.keys():
            if (method == "L_Shapley" and order != 1) or (method == "GraphSHAPIQ"):
                continue
            data_order = ORDERS[order]
            runtime_avg = float(data_order[f"runtime_{method}"].mean())
            mse_avg = float(data_order[f"mse_{method}"].mean())
            alpha = 0.7 if order != 1 else 1.0
            ax.scatter(
                runtime_avg,
                mse_avg,
                color=COLORS[method],
                marker=ORDER_MARKERS[order],
                edgecolors="black",
                s=SCATTER_SIZE,
                alpha=alpha,
                zorder=0 + z_ord,
            )

            # also add the informed second order data for all baseline methods as a hatched marker
            if order == 2 or order == 3:
                if order == 2:
                    data_order = data_second_order_informed
                else:
                    data_order = data_third_order_informed
                runtime_avg = float(data_order[f"runtime_{method}"].mean())
                mse_avg = float(data_order[f"mse_{method}"].mean())
                ax.scatter(
                    runtime_avg,
                    mse_avg,
                    color=COLORS[method],
                    marker=ORDER_MARKERS[order],
                    s=SCATTER_SIZE + 10,
                    edgecolors="black",
                    lw=2,
                    zorder=0 + z_ord,
                )
                ax.scatter(
                    runtime_avg,
                    mse_avg,
                    color=COLORS[method],
                    marker=ORDER_MARKERS[order],
                    hatch=5*"/",
                    s=SCATTER_SIZE,
                    edgecolors="white",
                    lw=0,
                    zorder=1 + z_ord,
                )

    # adjust the scale of the plot
    ax.set_xscale("log")
    ax.set_ylim(2e-3, 2e5)
    ax_graphsahpiq.set_ylim(-0.025, 0.04)
    ax.set_yscale("log")
    plt.minorticks_off()

    # plot graphshapiq
    for order in ORDERS.keys():
        data_order = ORDERS[order]
        runtime_avg = float(data_order["runtime_GraphSHAPIQ"].mean())
        mse_avg = 0.0
        ax_graphsahpiq.scatter(
            runtime_avg,
            mse_avg,
            color=COLORS["GraphSHAPIQ"],
            marker=ORDER_MARKERS[order],
            edgecolor="black",
            s=SCATTER_SIZE
        )
    ax_graphsahpiq.hlines(0, 0, 1000, colors="gray", linestyles="solid", lw=0.5, zorder=0)
    ax_graphsahpiq.set_yticks([0.0])
    ax_graphsahpiq.set_yticklabels(["exact"])

    # add a minature plot only of graphshapiq
    if add_inset:
        ax_inset = fig.add_axes([0.48, 0.191, 0.2, 0.09])
        for order in ORDERS.keys():
            data_order = ORDERS[order]
            runtime_avg = float(data_order["runtime_GraphSHAPIQ"].mean())
            mse_avg = 0.0
            ax_inset.plot(
                runtime_avg,
                mse_avg,
                color=COLORS["GraphSHAPIQ"],
                marker=ORDER_MARKERS[order],
                mec="black",
                markersize=MARKER_SIZE
            )
        ax_inset.set_yticks([])
        ax_inset.set_yticklabels([])
        ax_inset.set_xticks([1.9, 2.0])
        ax_inset.set_xticklabels(["1.9s", "2.0s"])
        ax_inset.set_xlim(1.86, 2.07)
        ax_inset.hlines(0, 0, 1000, colors="gray", linestyles="solid", lw=0.5, zorder=0)
        ax_inset.tick_params(axis='x', direction='in', pad=-15, labelsize=plt.rcParams["font.size"] - 6)

    # show where the inset came from and draw a rectangle around it in the main plot
    if add_inset:
        # add a polygon from the bottom left corner of the inset to the part of the main plot
        ax_graphsahpiq.add_patch(
            plt.Polygon(
                np.array([
                    [2.7, -0.013],
                    [4.9, -0.0059],
                    [5, -0.0059],
                    [5, 0.0339],
                    [4.77, 0.0339],
                    [2.7, 0.013],
                    [2.7, -0.013],
                    [1.5, -0.013],
                    [1.5, 0.013],
                    [2.7, 0.013],
                ]),
                closed=False,
                fill=False,
                lw=0.75,
                zorder=0,
                linestyle="--",
            )
        )

    # add grid only for x axis ticks for ax_graphsahpiq
    ax_graphsahpiq.grid(which="major", linestyle="--", color="gray", lw=0.5, axis="x")

    # add empty plots for the legend only with the order markers in color black
    for order in [1, 2, 3]:
        ax.plot(
            [],
            [],
            color="black",
            marker=ORDER_MARKERS[order],
            mec="black",
            markersize=MARKER_SIZE,
            label=ORDER_NAMES[order],
            lw=0
        )
    first_legend = ax.legend(title="$\\bf{Order}$", loc="upper left")

    # add graphshapiq name to legend as a second legend
    graphshapiq_legend = ax.plot(
        [],
        [],
        color=COLORS["GraphSHAPIQ"],
        marker="s",
        mec="black",
        markersize=MARKER_SIZE,
        label="GraphSHAP-IQ",
        lw=0
    )
    second_legend = ax.legend(handles=graphshapiq_legend, loc="upper right")
    ax.add_artist(second_legend)
    ax.add_artist(first_legend)

    ax_graphsahpiq.set_xlabel("Average Runtime in Seconds (log)")
    ax.set_ylabel("Average MSE (log)")
    # add title to the main plot
    ax.set_title("Mutagenicity with 2-layer GCN")

    # remove all white space between the plots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if add_inset:
        title = "mse_runtime_inset.pdf"
    else:
        title = "mse_runtime.pdf"
    plt.savefig(title)
    plt.show()
