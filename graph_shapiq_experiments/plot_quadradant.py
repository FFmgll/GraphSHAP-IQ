import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    add_inset = False
    plt.rcParams.update({"font.size": 20})  # increase the font size of the plot
    plt.rcParams["figure.figsize"] = (8, 7)
    MARKER_SIZE = 18

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

    data_first_order = pd.read_csv("GCN_Mutagenicity_2_1_0_runtime_metrics.csv")
    data_second_order = pd.read_csv("GCN_Mutagenicity_2_2_0_runtime_metrics.csv")
    data_third_order = pd.read_csv("GCN_Mutagenicity_2_3_0_runtime_metrics.csv")

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

    for method in methods:
        for order in ORDERS.keys():
            if (method == "L_Shapley" and order != 1) or (method == "GraphSHAPIQ"):
                continue
            data_order = ORDERS[order]
            runtime = data_order[f"runtime_{method}"]
            runtime_avg = float(runtime.mean())
            mse = data_order[f"mse_{method}"]
            mse_avg = float(mse.mean())
            # if method == "L_Shapley":  # TODO DELETE once the data is available
            #     runtime_avg = 1.978
            ax.plot(
                runtime_avg,
                mse_avg,
                color=COLORS[method],
                marker=ORDER_MARKERS[order],
                mec="white",
                markersize=MARKER_SIZE
            )

    # adjust the scale of the plot
    ax.set_xscale("log")
    ax.set_ylim(0.01, 10**4)
    ax.set_yscale("log")
    plt.minorticks_off()

    # plot graphshapiq
    for order in ORDERS.keys():
        data_order = ORDERS[order]
        runtime_avg = float(data_order["runtime_GraphSHAPIQ"].mean())
        mse_avg = 0.0
        ax_graphsahpiq.plot(
            runtime_avg,
            mse_avg,
            color=COLORS["GraphSHAPIQ"],
            marker=ORDER_MARKERS[order],
            mec="white",
            markersize=MARKER_SIZE
        )
    ax_graphsahpiq.hlines(0, 0, 1000, colors="gray", linestyles="solid", lw=1, zorder=0)
    ax_graphsahpiq.set_yticks([0.0])
    ax_graphsahpiq.set_yticklabels(["exact"])

    # add a minature plot only of graphshapiq
    if add_inset:
        ax_inset = fig.add_axes([0.415, 0.169, 0.3, 0.11])
        for order in ORDERS.keys():
            data_order = ORDERS[order]
            runtime_avg = float(data_order["runtime_GraphSHAPIQ"].mean())
            mse_avg = 0.0
            ax_inset.plot(
                runtime_avg,
                mse_avg,
                color=COLORS["GraphSHAPIQ"],
                marker=ORDER_MARKERS[order],
                mec="white",
                markersize=MARKER_SIZE
            )
        ax_inset.set_yticks([])
        ax_inset.set_yticklabels([])
        ax_inset.set_xlim(1.88, 2.053)
        ax_inset.hlines(0, 0, 1000, colors="gray", linestyles="solid", lw=1, zorder=0)
        ax_inset.tick_params(axis='x', direction='in', pad=-16, labelsize=plt.rcParams["font.size"] - 4)

    # show where the inset came from and draw a rectangle around it in the main plot
    if add_inset:
        # add a polygon from the bottom left corner of the inset to the part of the main plot
        ax_graphsahpiq.add_patch(
            plt.Polygon(
                np.array([
                    [2.39, -0.015],
                    [4.77, -0.0375],
                    [5, -0.0375],
                    [5, 0.0375],
                    [4.77, 0.0375],
                    [2.39, 0.015],
                    [2.39, -0.015],
                    [1.6, -0.015],
                    [1.6, 0.015],
                    [2.39, 0.015],
                ]),
                closed=False,
                fill=False,
                lw=0.75,
                zorder=0,
                linestyle="--",
            )
        )

    # add grid only on major ticks
    ax.grid(which="major", linestyle="--", color="gray", lw=0.5)
    ax_graphsahpiq.grid(which="major", linestyle="--", color="gray", lw=0.5)

    # add empty plots for the legend only with the order markers in color black
    for order in [1, 2, 3]:
        ax.plot(
            [],
            [],
            color="black",
            marker=ORDER_MARKERS[order],
            mec="white",
            markersize=MARKER_SIZE,
            label=ORDER_NAMES[order],
            lw=0
        )
    ax.legend(title="$\\bf{Order}$", loc="upper left")

    # add graphshapiq name to legend
    ax_graphsahpiq.plot(
        [],
        [],
        color=COLORS["GraphSHAPIQ"],
        marker="s",
        mec="white",
        markersize=MARKER_SIZE,
        label="GraphSHAPIQ",
        lw=0
    )
    ax_graphsahpiq.legend(loc="lower right")

    ax_graphsahpiq.set_xlabel("Average Runtime in Seconds (log)")
    ax.set_ylabel("Average MSE (log)")

    # remove all white space between the plots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if add_inset:
        title = "mse_runtime_inset.pdf"
    else:
        title = "mse_runtime.pdf"
    plt.savefig(title)
    plt.show()
