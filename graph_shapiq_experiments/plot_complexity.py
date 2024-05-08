import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def logarithmic_function(x, a, b):
    # Used to fit the trend curve
    return a * np.log(x) + b


def plot_trend_curve(x_values, values, color, type="lin"):
    if type == "log":
        popt, pcov = curve_fit(logarithmic_function, x_values, values)
        plt.plot(
            x_values,
            logarithmic_function(x_values, *popt),
            color=color,
            linestyle="--",
            linewidth=1,
            label="",
        )
    if type == "lin":
        coefficients = np.polyfit(x_values, values, 1)
        polynomial_func = np.poly1d(coefficients)
        plt.plot(
            x_values,
            polynomial_func(x_values),
            color=color,
            linestyle=":",
            linewidth=1,
            label="",
        )


def plot_naive_budget(node_range):
    plt.plot(
        node_range,
        node_range * np.log10(2),
        linestyle="--",
        linewidth=1,
        color="black",
        label="Naive Budget",
    )


def plot_complexity_by_layers(plot_dataset, dataset, scatter=False):
    COLORS = {
        "1": COLOR_LIST[0],
        "2": COLOR_LIST[1],
        "3": COLOR_LIST[2],
        "4": COLOR_LIST[3],
    }
    plt.rcParams.update(params)

    # Plot dataset
    fig, ax = plt.subplots()
    for n_layers in LAYERS:
        # Values to plot
        plot_dataset_layer = plot_dataset[plot_dataset["n_layers"] == n_layers]
        plot_medians = medians[dataset, "GCN", n_layers, :]
        plot_q1 = q1[dataset, "GCN", n_layers, :]
        plot_q3 = q3[dataset, "GCN", n_layers, :]
        # Labels for legend and colors
        plot_color = COLORS[n_layers]
        plot_label = n_layers + " conv. layers"
        # Fit and plot the logarithmic trend curve

        if scatter:
            plot_trend_curve(plot_medians.index,plot_medians, plot_color, type="log")
            # Plot the median as line plot
            ax.scatter(
                plot_dataset_layer["n_players"],
                plot_dataset_layer["log10_gSHAP_budget_capped"],
                color=plot_color,
                s=5,
                marker="s",
                label=plot_label,
            )
        else:
            plot_trend_curve(plot_medians.index,plot_medians, plot_color, type="log")
            # Plot the median as line plot
            ax.plot(
                plot_medians.index,
                plot_medians,
                color=plot_color,
                linestyle="-",
                label=plot_label,
            )
            # Plot the band around median using Q1 and Q3
            ax.fill_between(
                plot_medians.index,
                plot_q1,
                plot_q3,
                alpha=0.2,
                color=plot_color,
            )

    # Axis customization
    min_x = plot_dataset["n_players"].min()
    max_x = min(plot_dataset["n_players"].max(), 100)
    node_range = np.arange(min_x - min_x % 5, max_x, 5)
    plot_naive_budget(node_range)
    plt.xlim(min_x, max_x)
    plt.ylim(1, 10)
    plt.xticks(node_range)
    plt.yticks([2, 3, 4, 5, 6, 7, 8], ["100", "1k", "10k", "100k", "1m", "10m", "100m"])
    # Title and descriptions
    plt.ylabel("GraphSHAP-IQ Budget (in log10)")
    plt.xlabel("Number of Graph Nodes (n)")
    ax.legend()
    plt.title(dataset)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    if scatter:
        plt.savefig(os.path.join(save_path_plots, dataset + "_by_layers_scatter.png"))
    else:
        plt.savefig(os.path.join(save_path_plots, dataset + "_by_layers_median_q1_q3.png"))
    plt.show()


def plot_complexity_by_node_degree(plot_dataset, dataset, scatter=False, plot_by="avg_node_degree"):
    COLORS = {
        "1": COLOR_LIST[0],
        "2": COLOR_LIST[1],
        "3": COLOR_LIST[2],
        "4": COLOR_LIST[3],
    }
    plt.rcParams.update(params)

    # Plot dataset
    fig, ax = plt.subplots()
    for n_layers in LAYERS:
        # Values to plot
        plot_dataset_layer = plot_dataset[plot_dataset["n_layers"] == n_layers]
        plot_medians = medians[dataset, "GCN", n_layers, :]
        plot_q1 = q1[dataset, "GCN", n_layers, :]
        plot_q3 = q3[dataset, "GCN", n_layers, :]
        # Labels for legend and colors
        plot_color = COLORS[n_layers]
        plot_label = n_layers + " conv. layers"
        # Fit and plot the logarithmic trend curve

        if scatter:
            plot_trend_curve(
                plot_dataset_layer[plot_by],
                plot_dataset_layer["log10_gSHAP_budget_capped"],
                plot_color,
                type="lin",
            )
            # Plot the median as line plot
            ax.scatter(
                plot_dataset_layer[plot_by],
                plot_dataset_layer["log10_gSHAP_budget_capped"],
                color=plot_color,
                s=5,
                marker="s",
                label=plot_label,
            )

    # Axis customization
    min_x = plot_dataset[plot_by].min()
    max_x = min(plot_dataset[plot_by].max(), 100)
    node_range = np.arange(min_x - min_x % 5, max_x, 5)
    plt.xlim(min_x, max_x)
    plt.ylim(1, 10)
    plt.xticks(node_range)
    plt.yticks([2, 3, 4, 5, 6, 7, 8], ["100", "1k", "10k", "100k", "1m", "10m", "100m"])
    # Title and descriptions
    plt.ylabel("GraphSHAP-IQ Budget (in log10)")
    plt.xlabel(plot_by)
    ax.legend()
    plt.title(dataset)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_plots, dataset + "_by_layers_statistics.png"))
    plt.show()


if __name__ == "__main__":
    save_directory = "../results/complexity_analysis"
    save_path_plots = os.path.join(save_directory, "plots")
    results = {}

    COLOR_LIST = ["#ef27a6", "#7d53de", "#00b4d8", "#ff6f00"]
    params = {
        "legend.fontsize": "x-large",
        "figure.figsize": (7, 5),
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }

    DATASETS = [
        "AIDS",
        "DHFR",
        "COX2",
        "BZR",
        "PROTEINS",
        "ENZYMES",
        "MUTAG",
        "Mutagenicity",
    ]

    LAYERS = ["1", "2", "3", "4"]

    # Import dataset statistics
    dataset_statistics = {}
    for file_path in glob.glob(os.path.join(save_directory, "dataset_statistics", "*.csv")):
        dataset_statistic = pd.read_csv(file_path)
        dataset_name = file_path.split("/")[-1][:-4]
        if dataset_name in DATASETS:
            dataset_statistics[dataset_name] = dataset_statistic

    for file_path in glob.glob(os.path.join(save_directory, "*.csv")):
        result = pd.read_csv(file_path)
        file_name = file_path.split("/")[-1][:-4]  # remove path and ending .csv
        if file_name.split("_")[0] == "complexity":
            dataset_name = file_name.split("_")[2]
            result["model_type"] = file_name.split("_")[1]
            result["dataset_name"] = dataset_name
            result["n_layers"] = file_name.split("_")[3]
            result = pd.merge(
                result,
                dataset_statistics[dataset_name],
                left_index=True,
                right_index=True,
                how="inner",
            )
            results[file_name] = result

    df = pd.concat(results.values(), keys=results.keys())
    df["log10_gSHAP_budget"] = np.log10(df["exact_gSHAP"])
    df["log10_gSHAP_budget_capped"] = df["log10_gSHAP_budget"].clip(
        upper=df["n_players"] * np.log10(2)
    )
    df["n_players"] = df["n_players"].astype(int)
    df["log10_gSHAP_budget"] = np.log10(df["exact_gSHAP"])
    # df["player_bins"] = pd.cut(df["n_players"], bins=range(0, df["n_players"].max() + 6, 5), right=False)
    means = df.groupby(["dataset_name", "model_type", "n_layers", "n_players"])[
        "log10_gSHAP_budget"
    ].mean()
    q1 = df.groupby(["dataset_name", "model_type", "n_layers", "n_players"])[
        "log10_gSHAP_budget"
    ].quantile(0)
    q3 = df.groupby(["dataset_name", "model_type", "n_layers", "n_players"])[
        "log10_gSHAP_budget"
    ].quantile(1)
    medians = df.groupby(["dataset_name", "model_type", "n_layers", "n_players"])[
        "log10_gSHAP_budget"
    ].quantile(0.5)
    stds = df.groupby(["dataset_name", "model_type", "n_layers", "n_players"])[
        "log10_gSHAP_budget"
    ].std()


    for dataset in DATASETS:
        plot_dataset = df[df["dataset_name"] == dataset]
        #plot_complexity_by_layers(plot_dataset, dataset, scatter=True)
        plot_complexity_by_node_degree(
            plot_dataset, dataset, scatter=True, plot_by="graph_density"
        )
