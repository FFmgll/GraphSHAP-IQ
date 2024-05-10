"""This script is used to plot the explanation graphs for a given instance and explanaiton.

## Old Naming Convention (seperated by underscore):
    - Type of model, e.g. GCN, GIN
    - Dataset name, e.g. MUTAG
    - Number of Graph Convolutions, e.g. 2 graph conv layers
    - Graph bias, e.g. True, if the linear layer after global pooling has a bias
    - Node bias, e.g. True, if the convolution layers have a bias
    - Data ID: a technical identifier of the explained instance
    - Number of players, i.e. number of nodes in the graph
    - Largest neighborhood size as integer
"""


import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx

from shapiq.explainer.graph import get_explanation_instances
from shapiq.interaction_values import InteractionValues
from shapiq.plot.explanation_graph import explanation_graph_plot
from shapiq.moebius_converter import MoebiusConverter

RESULTS_DIR = os.path.join("..", "results", "single_gt_instances")
PLOT_DIR = os.path.join("..", "results", "explanation_graphs")
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


def load_file_into_interaction_values(path: str) -> InteractionValues:
    """Load the interaction values from a file."""
    df = pd.read_csv(path)
    # rename first col from "" to "set"
    df = df.rename(columns={df.columns[0]: "set"})

    values = []
    lookup = {}
    n_players = 0
    for i, row in df.iterrows():
        val = float(row["gt"])
        coalition = row["set"]
        if coalition == "()":
            coalition = tuple()
        else:
            coalition = coalition.replace("(", "").replace(")", "")
            coalition_members = coalition.split(",")
            coalition_transformed = []
            for member in coalition_members:
                if member == "" or member == " " or member == ",":
                    continue
                try:
                    member = int(member)
                except ValueError:
                    member = member
                coalition_transformed.append(member)
            coalition = tuple(coalition_transformed)
        lookup[coalition] = len(values) - 1
        values.append(val)
        n_players = max(n_players, max(coalition, default=0))
    values = np.array(values)
    n_players += 1

    example_values = InteractionValues(
        n_players=n_players,
        values=values,
        index="Moebius",
        interaction_lookup=lookup,
        baseline_value=float(values[lookup[tuple()]]),
        min_order=0,
        max_order=n_players,
    )
    return example_values


if __name__ == "__main__":

    results_file = "gtmoebius_GCN_MUTAG_3_True_True_4_11_11.csv"

    interaction_values = load_file_into_interaction_values(os.path.join(RESULTS_DIR, results_file))
    converter = MoebiusConverter(moebius_coefficients=interaction_values)
    k_sii_values = converter(index="k-SII", order=interaction_values.n_players)
    print(k_sii_values)

    dataset_name = results_file.split("_")[2]
    data_id = int(results_file.split("_")[6])

    explanation_instances = get_explanation_instances(dataset_name)
    graph_instance = explanation_instances[data_id]

    mutag_atom_names = ["C", "N", "O", "F", "I", "Cl", "Br"]
    # make one-hot encoding of the atom types into labels
    graph_labels = {
        node_id: mutag_atom_names[np.argmax(atom)]
        for node_id, atom in enumerate(graph_instance.x.numpy())
    }

    graph = to_networkx(graph_instance, to_undirected=True)

    _ = explanation_graph_plot(
        graph=graph,
        interaction_values=k_sii_values,
        plot_explanation=True,
        n_interactions=100,
        size_factor=1,
        compactness=5,
        random_seed=4,
        label_mapping=graph_labels,
        cubic_scaling=True,
    )
    plt.savefig(os.path.join(PLOT_DIR, f"{results_file}_graph_explanation.pdf"))
    plt.tight_layout()
    plt.show()

    # get the og graph
    _ = explanation_graph_plot(
        graph=graph,
        interaction_values=k_sii_values,
        plot_explanation=False,
        n_interactions=10,
        size_factor=5,
        compactness=25,
        random_seed=4,
        label_mapping=graph_labels,
    )
    plt.savefig(os.path.join(PLOT_DIR, f"{results_file}_graph.pdf"))
    plt.tight_layout()
    plt.show()
