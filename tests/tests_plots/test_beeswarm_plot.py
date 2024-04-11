"""This test module tests the beeswarm plot function in the shapiq package."""

import numpy as np
import pytest

from shapiq.utils import generate_interaction_lookup
from shapiq import InteractionValues
from shapiq.plot import beeswarm_plot


@pytest.mark.parametrize("max_order, baseline_value", [(1, 0.25), (1, 0), (2, 0.25)])
def test_beeswarm_plot(max_order, baseline_value):
    """Tests the basic functionality of the beeswarm plot function."""

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

    # test force plot
    fig, ax = beeswarm_plot(interaction_values)
    assert fig is not None
    assert ax is not None

    # TODO add more sensible tests
