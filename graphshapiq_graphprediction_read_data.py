import os
import glob
import pandas as pd
import numpy as np

from shapiq.approximator.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues

if __name__ == "__main__":
    save_directory = "results/single_gt_instances"
    save_path_plots = os.path.join(save_directory, "plots")
    moebius = {}

    GRAPH_PREDICTION_MODELS = [
        "AIDS",
        "DHFR",
        "COX2",
        "BZR",
        "PROTEINS",
        "ENZYMES",
        "MUTAG",
        "Mutagenicity",
    ]

    for file_path in glob.glob(os.path.join(save_directory, "*.csv")):

        results = pd.read_csv(file_path, index_col=0)
        file_name = file_path.split("/")[-1][:-4]  # remove path and ending .csv
        if file_name.split("_")[0] in ["gtmoebius","largemoebius"]:
            dataset_name = file_name.split("_")[2:4]
            n_layers = file_name.split("_")[3]
            moebius[file_name] = results


    sparsify_threshold = 10e-5
    INDEX_RANGE = np.arange(1,6)
    shapley_interactions = {}

    for id, data in moebius.items():
        shapley_interactions[id] = {}
        n_players = int(id.split("_")[7])
        interaction_lookup = {}
        for pos,interaction_tuple in enumerate(data.index):
            interaction_lookup[eval(interaction_tuple)] = pos
        moebius_interactions = InteractionValues(n_players=n_players,baseline_value=0,interaction_lookup=interaction_lookup,values=data["gt"].values,min_order=0,max_order=n_players,index="Moebius")
        moebius_converter = MoebiusConverter(moebius_interactions)
        for order in INDEX_RANGE:
            k_sii = moebius_converter.moebius_to_shapley_interaction(index="k-SII",order=order)
            k_sii.sparsify(sparsify_threshold)
            shapley_interactions[id][order] = k_sii