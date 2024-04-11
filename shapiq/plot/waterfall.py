"""This module contains the waterfall plot function for the shapiq package"""

import matplotlib.pyplot as plt

from shapiq import InteractionValues


def waterfall_plot(interaction_values: InteractionValues) -> tuple[plt.Figure, plt.Axes]:
    """Plots the attribution of each Shapley value/interaction as a waterfall plot.

    Note:
        This plot originates from the [`shap`](https://github.com/shap/shap/blob/master/shap/plots/_waterfall.py)
        library.

    Args:
        interaction_values: The interaction values.

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing the figure and the axis of the plot.
    """
    raise NotImplementedError("This function is not implemented yet.")
