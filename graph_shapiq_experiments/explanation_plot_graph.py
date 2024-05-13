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
from approximation_utils import EXACT_DIR, parse_file_name

PLOT_DIR = os.path.join("..", "results", "explanation_graphs")
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


if __name__ == "__main__":

    INDEX = "k-SII"

    MODEL_TYPE = "GCN"
    DATASET_NAME = "Mutagenicity"
    N_LAYER = 2
    DATA_ID = 197

    # plot parameter
    RANDOM_SEED = 4
    COMPACTNESS = 1.5
    SIZE_FACTOR = 2
    N_INTERACTIONS = 100
    CUBIC_SCALING = False

    file_identifier = "_".join([MODEL_TYPE, DATASET_NAME, str(N_LAYER), str(DATA_ID)])
    # find the file in EXACT_DIR that contains the identifier
    file_name = None
    DIRECTORIES = [EXACT_DIR, PLOT_DIR]
    for directory in DIRECTORIES:
        for file in os.listdir(directory):
            if file_identifier in file:
                file_name = file
                break
    if file_name is None:
        raise ValueError(f"File with identifier {file_identifier} not found.")

    plot_name = file_name.replace(".interaction_values", ".pdf")
    path_to_file = os.path.join(directory, file_name)

    moebius_values = InteractionValues.load(path_to_file)

    converter = MoebiusConverter(moebius_coefficients=moebius_values)

    attributes = parse_file_name(file_name)
    dataset_name = attributes["dataset_name"]
    data_id = attributes["data_id"]

    explanation_instances = get_explanation_instances(dataset_name)
    graph_instance = explanation_instances[data_id]

    mutag_atom_names = ["C", "N", "O", "F", "I", "Cl", "Br"]
    # make one-hot encoding of the atom types into labels
    graph_labels = {
        node_id: mutag_atom_names[np.argmax(atom)]
        for node_id, atom in enumerate(graph_instance.x.numpy())
    }
    graph = to_networkx(graph_instance, to_undirected=True)

    # plot full graph explanation ------------------------------------------------------------------
    all_k_sii_values = converter(index=INDEX, order=converter.n)
    print(all_k_sii_values)
    _ = explanation_graph_plot(
        graph=graph,
        interaction_values=all_k_sii_values,
        plot_explanation=True,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
    )

    # plot 2-SII explanation -----------------------------------------------------------------------
    k_sii_values = converter(index=INDEX, order=2)
    print(k_sii_values)
    _ = explanation_graph_plot(
        graph=graph,
        interaction_values=k_sii_values,
        plot_explanation=True,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
    )
    plt.savefig(os.path.join(PLOT_DIR, f"{plot_name}_graph_kSII_explanation.pdf"))
    plt.tight_layout()
    plt.show()

    # plot sv explanation --------------------------------------------------------------------------
    sv_values = converter(index=INDEX, order=1)
    print(sv_values)
    _ = explanation_graph_plot(
        graph=graph,
        interaction_values=sv_values,
        plot_explanation=True,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
    )
    plt.savefig(os.path.join(PLOT_DIR, f"{plot_name}_graph_SV_explanation.pdf"))
    plt.tight_layout()
    plt.show()

    # plot the og graph ----------------------------------------------------------------------------
    _ = explanation_graph_plot(
        graph=graph,
        interaction_values=all_k_sii_values,
        plot_explanation=False,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
    )
    plt.savefig(os.path.join(PLOT_DIR, f"{plot_name}_graph.pdf"))
    plt.tight_layout()
    plt.show()
