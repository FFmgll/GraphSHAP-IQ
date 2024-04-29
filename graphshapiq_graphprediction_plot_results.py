import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    save_directory = "results"
    save_path_plots = os.path.join(save_directory,"plots")
    for file_path in glob.glob(os.path.join(save_directory, "*.csv")):
        results = pd.read_csv(file_path)
        file_name = file_path.split("/")[-1][:-4] #remove path and ending .csv
        plot_approximation_quality(results,file_name,save_path_plots)