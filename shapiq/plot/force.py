"""This module contains the force plot function for the shapiq package"""

import matplotlib.pyplot as plt

from shapiq import InteractionValues


def force_plot(interaction_values: InteractionValues) -> tuple[plt.Figure, plt.Axes]:
    """Plots the attribution of each Shapley value/interaction as a force plot.

    Note:
        This plot originates from the [`shap`](https://github.com/shap/shap/blob/master/shap/plots/_force_matplotlib.py)
        library.

    Args:
        interaction_values: The interaction values.

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing the figure and the axis of the plot.
    """
    raise NotImplementedError("This function is not implemented yet.")
