"""This test module tests the force plot function in the shapiq package."""

import numpy as np
import pytest

from shapiq.utils import generate_interaction_lookup
from shapiq import InteractionValues
from shapiq.plot import force_plot


@pytest.mark.parametrize("max_order, baseline_value", [(1, 0.25), (1, 0), (2, 0.25)])
def test_force_plot(max_order, baseline_value):
    """Tests the basic functionality of the force plot function."""

    # test parameters
    n_features = 5
    min_order = 1
    index = "SV" if max_order == 1 else "k-SII"

    # create interaction values object
    interaction_lookup = generate_interaction_lookup(n_features, min_order, max_order)
    values = np.random.normal(0, 1, len(interaction_lookup))
    interaction_values = InteractionValues(
        values=values,
        index=index,
        max_order=max_order,
        min_order=min_order,
        n_players=n_features,
        interaction_lookup=interaction_lookup,
        baseline_value=baseline_value,
    )

    # test force plot
    fig, ax = force_plot(interaction_values)
    assert fig is not None
    assert ax is not None

    # TODO add more sensible tests
