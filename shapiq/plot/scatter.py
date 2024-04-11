"""This module contains the scatter plot function for the shapiq package"""

import matplotlib.pyplot as plt

from shapiq import InteractionValues


def scatter_plot(interaction_values: list[InteractionValues]) -> tuple[plt.Figure, plt.Axes]:
    """Plots a scatter plot of at most two features and colors the attributions. This plot only
        works for a collection of Shapley values (i.e. interactions of order 1).

    Note:
        This plot originates from the [`shap`](https://github.com/shap/shap/blob/master/shap/plots/_scatter.py)
        library.

    Args:
        interaction_values: The interaction values.

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing the figure and the axis of the plot.
    """
    raise NotImplementedError("This function is not implemented yet.")
