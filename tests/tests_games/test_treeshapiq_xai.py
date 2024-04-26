"""This test module tests the TreeSHAP-IQ explanation games."""

import numpy as np
import pytest

from shapiq.approximator.montecarlo import SHAPIQ
from shapiq.games import Game
from shapiq.games.benchmark import TreeSHAPIQXAI
from shapiq.utils.sets import powerset


@pytest.mark.parametrize("max_order", [2, 3])
@pytest.mark.parametrize("index", ["k-SII", "SII", "STII"])
def test_random_forest_selection(index, max_order, dt_clf_model, background_clf_dataset):
    """Tests the base TreeSHAP-IQ explanation game."""

    # start with classification model
    model = dt_clf_model
    data, target = background_clf_dataset
    x_explain = data[0]
    class_label = int(target[0])
    min_order = 1 if index not in ["STII", "FSII"] else max_order

    game = TreeSHAPIQXAI(
        x=x_explain,
        tree_model=model,
        index=index,
        class_label=class_label,
        max_order=max_order,
        min_order=min_order,
        normalize=False,
    )
    gt_interaction_values = game.gt_interaction_values

    assert isinstance(game, TreeSHAPIQXAI)
    assert isinstance(game, Game)
    assert game.n_players == len(x_explain)

    # get the test coalitions
    test_coalitions = np.array([game.empty_coalition, game.empty_coalition, game.grand_coalition])
    test_coalitions[1, 0] = True
    test_coalitions[1, 1] = True

    # test the value function
    worth = game.value_function(test_coalitions)
    assert worth.shape == (len(test_coalitions),)
    assert worth[0] == game.empty_value

    # test against the ground truth "approximation"
    n_players = game.n_players
    budget = 2**n_players
    top_order = index in ["STII", "FSII"]  # false for k-SII and SII
    approximator = SHAPIQ(n=n_players, max_order=max_order, index=index, top_order=top_order)
    estimates = approximator.approximate(budget=budget, game=game)
    assert estimates.estimation_budget <= budget and not estimates.estimated
    assert estimates.index == index

    for interaction in powerset(range(n_players), min_size=min_order, max_size=max_order):
        assert np.isclose(estimates[interaction], gt_interaction_values[interaction])


def test_adult():
    """Test the AdultCensus TreeSHAP-IQ explanation game."""
    raise NotImplementedError("TODO: Implement this test!")


def test_california():
    """Test the CaliforniaHousing TreeSHAP-IQ explanation game."""
    raise NotImplementedError("TODO: Implement this test!")


def test_bike():
    """Test the BikeSharing TreeSHAP-IQ explanation game."""
    raise NotImplementedError("TODO: Implement this test!")
