"""This test module tests the scatter plot function in the shapiq package."""

import numpy as np
import pytest

from shapiq.utils import generate_interaction_lookup
from shapiq import InteractionValues
from shapiq.plot import scatter_plot


@pytest.mark.parametrize("max_order", [1, 2])
def test_scatter_plot(max_order):
    """Tests the basic functionality of the bar importance plot function."""

    # test parameters
    n_explanations = 10
    n_features = 5
    min_order = 1
    baseline_value = 0.25
    index = "SV" if max_order == 1 else "k-SII"

    # create interaction values objects
    interaction_lookup = generate_interaction_lookup(n_features, min_order, max_order)
    interaction_values = [
        InteractionValues(
            values=np.random.normal(0, 1, len(interaction_lookup)),
            index=index,
            max_order=max_order,
            min_order=min_order,
            n_players=n_features,
            interaction_lookup=interaction_lookup,
            baseline_value=baseline_value,
        )
        for _ in range(n_explanations)
    ]

    # if max_order >= 2 test for ValueError
    if max_order >= 2:
        with pytest.raises(ValueError):
            scatter_plot(interaction_values)
        return  # finish test here

    # test force plot
    fig, ax = scatter_plot(interaction_values)
    assert fig is not None
    assert ax is not None

    # TODO add more sensible tests
