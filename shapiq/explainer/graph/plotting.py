"""This module contains the class MutagPlotter, which is used to quickly plot graphs and boxplots
given instances of the MUTAG dataset."""

import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import pandas as pd

ATOM_NAMES = ["C", "N", "O", "F", "I", "Cl", "Br"]


def plot_graph(graph_instance, masked_nodes, save_path, random_state=None) -> None:
    """Plots the graph with the masked nodes highlighted in grey.

    Args:
        graph_instance: The graph instance to be plotted.
        masked_nodes: The nodes that are masked in the graph.
        save_path: The path where the plot should be saved.
        random_state: The random state for the layout of the graph. Defaults to None.
    """
    fig, axis = plt.subplots(1, 1)
    g = to_networkx(graph_instance, to_undirected=True)  # converts into graph
    pos = nx.spring_layout(g, seed=random_state)  # get positions of the nodes
    node_colors = ["lightblue" if node not in masked_nodes else "lightgrey" for node in g.nodes()]
    labels = {n: str(n) for n in range(graph_instance.num_nodes)}
    nx.draw(
        g, ax=axis, pos=pos, node_color=node_colors, node_shape="o", labels=labels, linewidths=2
    )
    plt.savefig(save_path)
    plt.show()


def boxplot_by_size(values, sizes, title) -> None:
    """Creates a boxplot based on values and grouped by sizes.

    Args:
        values: The values to be plotted.
        sizes: The sizes to group the values by.
        title: The title of the plot.
    """
    df_values = pd.DataFrame(values, columns=["values"])
    df_values["set_sizes"] = sizes
    # plot a boxplot of moebius values for each of the set sizes next toeach other
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    df_values.boxplot(column="values", by="set_sizes", ax=ax)
    ax.set_xlabel("Set Size")
    ax.set_ylabel("Value")
    # remove title
    plt.suptitle(title)
    ax.set_title("Values for Different Set Sizes")
    plt.show()


def _add_labels_to_plot(graph, ax, pos, sizes):
    """Adds the node labels to the plot."""
    labels = {n: ATOM_NAMES[int(graph.x[n].argmax())] for n in range(graph.num_nodes)}
    # add the labels as text to the plot with a small offset to the nodes below
    for node, (x, y) in pos.items():
        # get size of the node
        size = sizes[node]
        # change offset depending on size
        offset = 0.05 + (size / 1000) * 0.1
        ax.text(x, y + offset, labels[node], ha="center", fontsize=10)


class MutagPlotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_graph(graph_instance, masked_nodes, save_path, random_state=None):
        plot_graph(graph_instance, masked_nodes, save_path, random_state)

    @staticmethod
    def boxplot_by_size(values, sizes, title):
        boxplot_by_size(values, sizes, title)
