"""This script is used to illustrate the graph-based SHAP-IQ method. It randomly selects 5
instances from the MUTAG dataset and masks 1/3 of the nodes in each instance.
The masked nodes are highlighted in red. The graph with masked nodes is saved in the
results/illustration folder."""

import os

import numpy as np

from shapiq.explainer.graph.plotting import MutagPlotter
from shapiq.explainer.graph import get_explanation_instances


if __name__ == "__main__":
    plotter = MutagPlotter()
    explanation_instances = get_explanation_instances("MUTAG")

    save_path = os.path.join("results", "illustration")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    counter = 0
    for data_id, instance in enumerate(explanation_instances):
        if instance.num_nodes >= 20:
            if counter >= 5:
                continue
            masked_nodes = np.random.choice(
                range(instance.num_nodes), size=int(instance.num_nodes / 3), replace=False
            )
            plotter.plot_graph(
                instance, masked_nodes, save_path + "/" + str(data_id) + ".png", random_state=42
            )
            counter += 1
