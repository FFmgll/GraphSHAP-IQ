import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.special import binom
from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset
import random


class GraphPlotter:
    def __init__(self):
        self.RED = "#ff0d57"  # for positive attributions
        self.BLUE = "#1e88e5"  # for negative attributions
        self.LINE_COLOR = "#cccccc"  # for the color of the edges
        self.ATOM_NAMES = [
            "C",
            "N",
            "O",
            "F",
            "I",
            "Cl",
            "Br",
        ]  # the names of the atoms the features are one-hot encoded based on these names

    def plot_graph(self, graph_instance, masked_nodes, save_path):
        # Plot the graphs with Networkx
        plt.figure()
        g = to_networkx(graph_instance, to_undirected=True)

        node_colors = [
            "lightblue" if node not in masked_nodes else "lightgrey"
            for node in g.nodes()
        ]

        labels = {n: str(n) for n in range(graph_instance.num_nodes)}
        ax1 = plt.plot()
        nx.draw(g, node_color=node_colors, node_shape="o", labels=labels, linewidths=2)
        plt.savefig(save_path)
        plt.show()

    def boxplot_by_size(self, values, sizes, title):
        # Creates a boxplot based on values and grouped by sizes
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

    def _add_labels_to_plot(self, graph, ax, pos, sizes):
        """
        Scales and adds the atom names as labels to the plot.
        :param graph: the graph to plot.
        :param ax: the axis to plot on.
        :param pos: the positions of the nodes.
        :param sizes: the sizes of the nodes.
        """
        # get atom names as labels
        labels = {n: self.ATOM_NAMES[int(graph.x[n].argmax())] for n in range(graph.num_nodes)}
        # add the labels as text to the plot with a small offset to the nodes below
        for node, (x, y) in pos.items():
            # get size of the node
            size = sizes[node]
            # change offset depending on size
            offset = 0.05 + (size / 1000) * 0.1
            ax.text(x, y + offset, labels[node], ha="center", fontsize=10)

    def plot_graph_with_sv(self, graph, sv: np.ndarray, axis=None):
        """
        Plots the graph/molekule with the Shapley values as colors and sizes of the nodes.
        A high positive Shapley value results in a large, red node, a high negative Shapley value in a large, blue node.
        A small Shapley value results in a small node.
        :param graph: the graph to plot.
        :param sv: the Shapley values to plot.
        :param ax: the axis to plot on.
        """
        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=(5, 5))
        else:
            fig = axis.get_figure()

        # fill in the graph
        g = to_networkx(graph, to_undirected=True)
        pos = nx.spring_layout(g)

        # size up the nodes
        sizes = 1000 * np.abs(sv)

        # define colors
        colors = [self.RED if s > 0 else self.BLUE for s in sv]

        self._add_labels_to_plot(graph, axis, pos, sizes)

        # draw the graph
        nx.draw(g, pos, ax=axis, node_size=sizes, node_color=colors, edge_color=self.LINE_COLOR)

        return fig, axis

    def plot_graph_with_ksii(
        self, graph, interaction_scores: InteractionValues, axis=None, edge_threshold: float = 0.01
    ):
        """
        Plots the graph/molekule with the k-SII values as colors and sizes of the nodes and edges.
        First order k-SII values are represented as node sizes, second order k-SII values as edge sizes.
        A high positive k-SII value results in a large, red node/edge, a high negative k-SII value in a large, blue node/edge.
        A small k-SII value results in a small node/edge.
        :param graph: the graph to plot.
        :param first_order: the first order k-SII values to plot.
        :param second_order: the second order k-SII values to plot.
        :param axis: the axis to plot on.
        :param edge_threshold: the threshold for the edge values to be plotted.
        """

        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=(5, 5))
        else:
            fig = axis.get_figure()

        # fill in the graph
        g = to_networkx(graph, to_undirected=True)
        pos = nx.spring_layout(g)

        first_order = np.zeros(len(g.nodes))

        for i, interaction in enumerate(powerset(g.nodes, min_size=1, max_size=1)):
            first_order[i] = interaction_scores[interaction]

        # get all original edges
        og_edges: list[tuple[int, int]] = g.edges

        # change nodes
        sizes = 1000 * np.abs(first_order)
        colors = [self.RED if s > 0 else self.BLUE for s in first_order]

        # add labels to the plot
        self._add_labels_to_plot(graph, axis, pos, sizes)

        # create edges
        edge_widths = []
        edge_colors = []
        seen_edges: list[tuple[int, int]] = []
        # add the edges between all pairs of nodes to the graph
        for i in g.nodes:
            for j in g.nodes:
                edge = tuple(sorted([i, j]))
                if i == j or edge in seen_edges:
                    continue
                interaction_score = interaction_scores[edge]
                if abs(interaction_score) < edge_threshold:
                    continue

                g.add_edge(i, j)
                edge_widths.append(abs(interaction_score) * 30)
                edge_colors.append(self.RED if interaction_score > 0 else self.BLUE)
                seen_edges.append(edge)

        # max_width = max(edge_widths)
        edge_alphas = [0.4 for _ in seen_edges]

        # draw the graph
        nx.draw(g, pos, ax=axis, node_size=sizes, node_color=colors, edge_color=self.LINE_COLOR)

        # add the interaction edges
        nx.draw_networkx_edges(
            g,
            pos,
            ax=axis,
            edgelist=seen_edges,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=edge_alphas,
        )

        return fig, axis
