"""This script is used to summarize the current state of the approximation results. It combines all
approximation results into one dataframe and prints summary statistics of the experiment."""
import itertools
import os

from approximation_utils import create_results_overview_table

if __name__ == "__main__":

    results_df = create_results_overview_table()
    total_size_on_disk = 0
    for dataset_name, model_id, n_layer in itertools.product(
        results_df["dataset_name"].unique(),
        results_df["model_id"].unique(),
        results_df["n_layers"].unique(),
    ):
        selection = results_df[
            (results_df["dataset_name"] == dataset_name)
            & (results_df["model_id"] == model_id)
            & (results_df["n_layers"] == n_layer)
        ]
        if selection.empty:
            continue
        print(f"Dataset: {dataset_name}, Model: {model_id}, Layers: {n_layer}")
        print(f"Number of instances: {len(selection)}")
        for small_graph in [True, False]:
            selection_setting = selection[selection["small_graph"] == small_graph]
            n_exact_small = len(selection_setting[selection_setting["exact"] == True])
            print(f"Number of exact instances (small graph: {small_graph}): {n_exact_small}")
            approx_results, approx_size_on_disk = {}, {}
            for approx in selection_setting["approximation"].unique():
                if approx == "exact":
                    continue
                selection_approx = selection_setting[selection_setting["approximation"] == approx]
                n_approx = len(selection_approx)
                approx_results[approx] = n_approx
                # get the size of the approximations on disk
                approx_size = 0
                for _, row in selection_approx.iterrows():
                    size = os.path.getsize(row["file_path"])
                    approx_size += size / 1024 / 1024  # in MB
                approx_size_on_disk[approx] = approx_size
                total_size_on_disk += approx_size
            approx_str = ", ".join([f"{k}: {v}" for k, v in approx_results.items()])
            print(f"Number of approximations: {approx_str}")
            approx_size_str = ", ".join(
                [f"{k}: {v:.2f} MB" for k, v in approx_size_on_disk.items()]
            )
            print(f"Size of approximations on disk: {approx_size_str}")
        print()

    print(f"Total size of approximations on disk: {total_size_on_disk:.2f} MB")
