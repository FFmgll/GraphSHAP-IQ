"""This module contains the plot function for the bar importance scores."""

import matplotlib.pyplot as plt

from shapiq import InteractionValues


def bar_importance_plot(interaction_values: list[InteractionValues]) -> tuple[plt.Figure, plt.Axes]:
    """Plots the mean absolute value of the Shapley interactions or attributions for each feature
        as a bar plot of the features' importances.

    Note:
        This plot originates from the [`shap`](https://github.com/shap/shap/blob/master/shap/plots/_bar.py)
        library.

    Args:
        interaction_values: The interaction values.

    Returns:
        A tuple containing the figure and the axis of the plot.
    """
    raise NotImplementedError("This function is not implemented yet.")
