"""This module contains all benchmark games for the feature selection setting."""

from typing import Optional

from shapiq.games.benchmark._setup import GameBenchmarkSetup

from .base import FeatureSelection


class AdultCensus(FeatureSelection):
    """The Adult Census dataset as a Feature Selection benchmark game.

        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree', 'random_forest', and 'gradient_boosting'.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
        random_state: The random state to use for the imputer. Defaults to 42.
    """


    def __init__(
            self,
            model_name: str = "decision_tree",
            normalize: bool = True,
            verbose: bool = True,
            random_state: Optional[int] = 42,
        ) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="adult_census",
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )

        super().__init__(
            x_train=setup.x_train,
            x_test=setup.x_test,
            y_train=setup.y_train,
            y_test=setup.y_test,
            fit_function=setup.fit_function,
            score_function=setup.score_function,
            normalize=normalize,
        )


class BikeSharing(FeatureSelection):
    """The Bike Sharing dataset as a Feature Selection benchmark game.

    Args:
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree', 'random_forest', and 'gradient_boosting'.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
        random_state: The random state to use for the imputer. Defaults to 42.
    """

    def __init__(
            self,
            model_name: str = "decision_tree",
            normalize: bool = True,
            verbose: bool = True,
            random_state: Optional[int] = 42,
        ) -> None:
        setup = GameBenchmarkSetup(
            dataset_name="bike_sharing",
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )

        super().__init__(
            x_train=setup.x_train,
            x_test=setup.x_test,
            y_train=setup.y_train,
            y_test=setup.y_test,
            fit_function=setup.fit_function,
            score_function=setup.score_function,
            normalize=normalize,
        )


class CaliforniaHousing(FeatureSelection):
    """The California Housing dataset as a Feature Selection benchmark game.

    Args:
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree', 'random_forest', and 'gradient_boosting'.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
        random_state: The random state to use for the imputer. Defaults to 42.
    """

    def __init__(
            self,
            model_name: str = "decision_tree",
            normalize: bool = True,
            verbose: bool = True,
            random_state: Optional[int] = 42,
        ) -> None:
        setup = GameBenchmarkSetup(
            dataset_name="california_housing",
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )

        super().__init__(
            x_train=setup.x_train,
            x_test=setup.x_test,
            y_train=setup.y_train,
            y_test=setup.y_test,
            fit_function=setup.fit_function,
            score_function=setup.score_function,
            normalize=normalize,
        )


# Path: shapiq/games/benchmark/feature_selection/benchmark.py
