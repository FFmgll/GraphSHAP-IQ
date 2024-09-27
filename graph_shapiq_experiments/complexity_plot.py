import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def logarithmic_function(x, a, b):
    # Used to fit the trend curve
    return a * np.log(x) + b


def budget_label():
    return "Number of Model Calls (in log 10)"


def node_label():
    return "Number of Graph Nodes (n)"


def plot_trend_curve(x_values, values, color, type="log"):
    x_val = np.array(x_values)
    y_val = np.array(values)

    if type == "log":
        popt, pcov = curve_fit(logarithmic_function, x_val, y_val)
        y_fit = logarithmic_function(x_val, *popt)
        #
        idx_to_plot = y_fit < x_val * np.log10(2)
        plot_y = y_fit[idx_to_plot]
        plot_x = x_val[idx_to_plot]

        plt.plot(
            plot_x,
            plot_y,
            color=color,
            linestyle="--",
            linewidth=1,
            label="",
        )
    if type == "lin":
        coefficients = np.polyfit(x_val, y_val, 1)
        polynomial_func = np.poly1d(coefficients)
        plt.plot(
            x_val,
            polynomial_func(x_val),
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
        label="Baseline",
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
        plot_medians = medians[dataset, n_layers, :]
        plot_q1 = q1[dataset, n_layers, :]
        plot_q3 = q3[dataset, n_layers, :]
        # Labels for legend and colors
        plot_color = COLORS[n_layers]
        if n_layers == "1":
            plot_label = n_layers + " Layer"
        else:
            plot_label = n_layers + " Layers"
        # Fit and plot the logarithmic trend curve

        if scatter:
            try:
                plot_trend_curve(plot_medians.index, plot_medians, plot_color, type="log")
            except:
                print("No trend curve fitted")
            # Plot the median as line plot
            ax.scatter(
                plot_dataset_layer["n_players"],
                plot_dataset_layer["log10_budget"],
                color=plot_color,
                s=5,
                marker="s",
                label=plot_label,
            )
        else:
            try:
                plot_trend_curve(plot_medians.index, plot_medians, plot_color, type="log")
            except:
                print("No trend curve fitted")
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
    min_x = plot_dataset["n_players"].min() - 2
    max_x = min(plot_dataset["n_players"].max() + 2, 150)

    if max_x > 40:
        legend_loc = "lower right"
    else:
        legend_loc = "upper left"
    node_range = np.arange(min_x - min_x % 5, max_x, max_x // 10)
    plot_naive_budget(node_range)
    plt.xlim(min_x, max_x)
    plt.ylim(1, 8.5)
    plt.xticks(node_range)
    plt.yticks([2, 3, 4, 5, 6, 7, 8], ["100", "1k", "10k", "100k", "1m", "10m", "100m"])
    # Title and descriptions
    plt.ylabel(budget_label())
    plt.xlabel(node_label())
    ax.legend(loc=legend_loc)
    plt.title(dataset)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    if scatter:
        plt.savefig(os.path.join(save_path_plots, "complexity_by_layers_" + dataset + ".png"))
    else:
        plt.savefig(
            os.path.join(save_path_plots, "complexity_by_layers_" + dataset + "_median_q1_q3.png")
        )
    plt.show()


def plot_complexity_by_statistic(
    save_id, plot_dataset, statistic, title, clabel, min_x, max_x, min_v, max_v, min_y, max_y, cmap
):
    plt.rcParams.update(params)

    # Plot dataset
    fig, ax = plt.subplots()
    # Plot the median as line plot
    scatter = ax.scatter(
        x=plot_dataset["n_players"],
        y=plot_dataset["log10_budget"],
        c=plot_dataset[statistic],
        cmap=cmap,
        s=5,
        marker="s",
        vmin=min_v,
        vmax=max_v,
    )
    if scatter.get_offsets().size > 0:
        plt.colorbar(scatter, label=clabel)  # Add a colorbar if data is present
    else:
        print("No data to create colorbar.")
    # Axis customization
    node_range = np.arange(min_x - min_x % 5, max_x, 10)
    plot_naive_budget(node_range)
    plt.xlim(min_x, max_x)
    plt.xticks(node_range)
    plt.yticks([2, 3, 4, 5, 6, 7, 8], ["100", "1k", "10k", "100k", "1m", "10m", "100m"])
    plt.ylim(min_y, max_y)
    # Title and descriptions
    plt.ylabel(budget_label())
    plt.xlabel(node_label())
    ax.legend()
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_plots, save_id + ".png"))
    plt.show()


if __name__ == "__main__":
    save_directory = "../results/complexity_analysis"
    save_path_plots = os.path.join(save_directory, "plots")
    results = {}

    COLOR_LIST = ["#ef27a6", "#7d53de", "#00b4d8", "#ff6f00", "#ffba08"]
    params = {
        #"legend.fontsize": "x-large",
        "figure.figsize": (8, 7),
        #"axes.labelsize": "x-large",
        #"axes.titlesize": "x-large",
        #"xtick.labelsize": "x-large",
        #"ytick.labelsize": "x-large",
        "font.size": 20
    }

    #plt.rcParams.update({"font.size": 20})  # increase the font size of the plot
    #plt.rcParams["figure.figsize"] = (8, 7)  # set figure size

    DATASETS = [
        "COX2",
        "BZR",
        "PROTEINS",
        "ENZYMES",
        "Mutagenicity",
        "FluorideCarbonyl",
        "Benzene",
        "AlkaneCarbonyl",
        "WaterQuality"
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
            dataset_name = file_name.split("_")[1]
            if dataset_name in DATASETS:
                result["dataset_name"] = dataset_name
                result["n_layers"] = file_name.split("_")[2]
                result = pd.merge(
                    result,
                    dataset_statistics[dataset_name],
                    left_index=True,
                    right_index=True,
                    how="inner",
                )
                results[file_name] = result

    all_datasets = pd.concat(dataset_statistics.values(), keys=dataset_statistics.keys())

    # Rename unnamed index column and introduce dataset_name column
    all_datasets["dataset_name"] = all_datasets.index.get_level_values(0)
    # Compute counts of graphs and means of graph statistics
    dataset_counts = all_datasets.groupby(["dataset_name"]).count()
    dataset_means = all_datasets.groupby(["dataset_name"]).mean()
    # Round and transform to percentages for better readability
    dataset_means["avg_num_nodes"] =  np.round(dataset_means["0"],2)
    dataset_means["avg_graph_density"] = np.round(dataset_means["graph_density"]*100,2)

    df = pd.concat(results.values(), keys=results.keys())
    df["log10_budget"] = np.log10(df["budget"].astype(float))
    df["log10_budget_capped"] = df["log10_budget"].clip(upper=df["n_players"] * np.log10(2))

    means = df.groupby(["dataset_name", "n_layers", "n_players"])["log10_budget"].mean()
    q1 = df.groupby(["dataset_name", "n_layers", "n_players"])["log10_budget"].quantile(0.25)
    q3 = df.groupby(["dataset_name", "n_layers", "n_players"])["log10_budget"].quantile(0.75)
    medians = df.groupby(["dataset_name", "n_layers", "n_players"])["log10_budget"].median()
    stds = df.groupby(["dataset_name", "n_layers", "n_players"])["log10_budget"].std()


    # Set global display format to suppress scientific notation
    pd.options.display.float_format = '{:.0e}'.format

    # We compute budget ratios in log-scale for numerical stability and report median of those values
    df["budget_ratio_perc"] = np.exp((np.log(df["budget"].astype(float)) - df["n_players"]*np.log(2)))*100
    budget_ratio_perc_median = np.round(df.groupby(["dataset_name", "n_layers"])["budget_ratio_perc"].median(),4)
    # Compute the multiplier reported in Table 1.
    budget_speedup_multiplier = np.round(100/df.groupby(["dataset_name", "n_layers"])["budget_ratio_perc"].median(),0)

    for dataset in df["dataset_name"].unique():
        if dataset != "WaterQuality":
            # Do not plot Water quality
            # Plots the dataset with a scatter plot and a line plot (median) with bands (Q1,Q3)
            plot_dataset = df[df["dataset_name"] == dataset]
            plot_complexity_by_layers(plot_dataset, dataset, scatter=True)
            plot_complexity_by_layers(plot_dataset, dataset, scatter=False)

    # Graph Density Plot
    dataset_name = "Mutagenicity"
    n_layers = "2"
    plot_dataet = df.copy()
    plot_dataset = df[df["n_layers"] == n_layers]
    plot_dataset = plot_dataset[plot_dataset["dataset_name"] == dataset_name]
    save_id = "complexity_by_graph_density_" + dataset_name + "_" + n_layers
    min_x = max(plot_dataset["n_players"].min(), 5)
    max_x = min(plot_dataset["n_players"].max(), 65)
    min_v = 0.05
    max_v = 0.55
    min_y = 1.5
    max_y = 6.5
    statistic = "graph_density"
    clabel = "Graph Density"
    title = "Mutagenicity (2-Layer GNN)"#"Exact Shapley Explanations on Mutagenicity (2-Layer GNN)"
    plot_complexity_by_statistic(
        save_id,
        plot_dataset,
        statistic,
        title,
        clabel,
        min_x,
        max_x,
        min_v,
        max_v,
        min_y,
        max_y,
        cmap="plasma",
    )


    # Graph Density Plot
    dataset_name = "Mutagenicity"
    n_layers = "2"
    plot_dataet = df.copy()
    plot_dataset = df[df["n_layers"] == n_layers]
    plot_dataset = plot_dataset[plot_dataset["dataset_name"] == dataset_name]
    save_id = "complexity_by_max_degree_" + dataset_name + "_" + n_layers
    min_x = max(plot_dataset["n_players"].min(), 5)
    max_x = min(plot_dataset["n_players"].max(), 65)
    min_v = 10
    max_v = 65
    min_y = 1.5
    max_y = 6.5
    statistic = "max_node_degree"
    clabel = "Maximum Node Degree"
    title = "Exact Shapley Explanations on Mutagenicity (2-Layer GNN)"
    plot_complexity_by_statistic(
        save_id,
        plot_dataset,
        statistic,
        title,
        clabel,
        min_x,
        max_x,
        min_v,
        max_v,
        min_y,
        max_y,
        cmap="plasma",
    )
