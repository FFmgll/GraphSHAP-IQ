import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_approximation_quality(results,file_name,save_path_plots):
    APPROXIMATORS = ["GraphSHAPIQ_False_interaction","GraphSHAPIQ_True_interaction","Permutation","KernelSHAPIQ","incKernelSHAPIQ","SHAPIQ","SVARMIQ"]
    plt.figure()
    n = file_name.split("_")[-2]
    for approximator in APPROXIMATORS:
        plt.plot(results["budget_with_efficiency"].values/2**int(n)*100,np.log(results[approximator].values),label=approximator)
    plt.legend()
    plt.title(file_name)
    plt.savefig(os.path.join(save_path_plots,(file_name+".png")))
    plt.show()


def plot_complexity(results,file_name,save_path_plots):
    colors = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf',
              '#1a9850', '#a6cee3', '#66a61e', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a',
              '#ffff99']

    BASELINES = [1e4,1e5,1e6,1e7,1e8]

    plt.figure()
    for i,n_layers in enumerate(results):
        results_current_layer = results[n_layers]
        results_current_layer_est = results_current_layer[results_current_layer["budget_estimated"]==True]
        results_current_layer_exact = results_current_layer[results_current_layer["budget_estimated"]==False]
        max_nodes = results_current_layer["n_players"].max()
        min_nodes = results_current_layer["n_players"].min()
        #results[n_layers] = results[n_layers][results[n_layers]["budget_estimated"]==True]
        #plt.scatter(results_current_layer["n_players"],results_current_layer["log10_budget_ratio_perc"],color=colors[i],s=5, label="GCN " + n_layers+" Layers")
        plt.scatter(results_current_layer_exact["n_players"], results_current_layer_exact["log10_budget_ratio_perc"], marker="^", color=colors[i],
                s=5, label="GCN" + n_layers + "- exact")
        plt.scatter(results_current_layer_est["n_players"],results_current_layer_est["log10_budget_ratio_perc"],color=colors[i],s=5, marker="s", label="GCN" + n_layers+"- est")

    node_range = np.arange(min_nodes,max_nodes)
    for baseline_budget in BASELINES:
        plt.plot(node_range,np.log10(baseline_budget)+2-node_range*np.log10(2), linestyle="--", linewidth=1,color="darkgrey", label=str("{:,}".format(int(baseline_budget)))+" budget")
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.legend(loc="lower left")
    plt.title(file_name)
    plt.savefig(os.path.join(save_path_plots,(file_name+".png")))
    plt.show()


if __name__ == "__main__":
    save_directory = "results"
    save_path_plots = os.path.join(save_directory,"plots")
    complexity_results = {}
    for file_path in glob.glob(os.path.join(save_directory, "*.csv")):
        results = pd.read_csv(file_path)
        file_name = file_path.split("/")[-1][:-4] #remove path and ending .csv
        if file_name.split("_")[0] == "complexity":
            dataset_name = file_name.split("_")[2]
            n_layers = file_name.split("_")[3]
            if dataset_name in complexity_results and dataset_name in ["MUTAG","PROTEINS","ENZYMES"]:
                complexity_results[dataset_name][n_layers] = results
            else:
                complexity_results[dataset_name] = {}
                complexity_results[dataset_name][n_layers] = results

        else:
            print("")
            #plot_approximation_quality(results,file_name,save_path_plots)

    for dataset_name in complexity_results:
        if dataset_name in ["ENZYMES","PROTEINS"]:
            plot_complexity(complexity_results[dataset_name], dataset_name, save_path_plots)