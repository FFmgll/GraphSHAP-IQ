"""This module contains the base class for the ensemble selection games."""

from typing import Callable, Optional

import numpy as np
from utils import Model

from ...base import Game


class EnsembleSelection(Game):
    """The Ensemble Selection game.

    The ensemble selection game models ensemble selection problems as cooperative games. The players
    are ensemble members and the value of a coalition is the performance of the ensemble on a
    test set.

    Note:
        Depending on the ensemble members, this game requires the `sklearn` and `xgboost` packages.

    Args:
        x_train: The training data as a numpy array of shape (n_samples, n_features).
        y_train: The training labels as a numpy array of shape (n_samples,).
        x_test: The test data as a numpy array of shape (n_samples, n_features).
        y_test: The test labels as a numpy array of shape (n_samples,).
        loss_function: The loss function to use for the ensemble members as a callable expecting
            two arguments: y_true and y_pred and returning a float.
        dataset_type: The type of dataset. Available dataset types are 'classification' and
            'regression'. Defaults to 'classification'.
        ensemble_members: A optional list of ensemble members to use. Defaults to None. If None,
            then the ensemble members are determined by the game. Available ensemble members are
            - 'regression' (will use linear regression for regression datasets and logistic
                regression for classification datasets)
            - 'decision_tree'
            - 'random_forest'
            - 'gradient_boosting'
            - 'knn'
            - 'svm'
        n_members: The number of ensemble members to use. Defaults to 10. Ignored if
            `ensemble_members` is not None.
        verbose: Whether to print information about the game and the ensemble members. Defaults to
            True.
        random_state: The random state to use for the ensemble members. Defaults to None.
        normalize: Whether to normalize the game values. Defaults to True.
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        loss_function: Callable[[np.ndarray, np.ndarray], float],
        dataset_type: str = "classification",
        ensemble_members: Optional[list[str]] = None,
        n_members: int = 10,
        verbose: bool = True,
        random_state: Optional[int] = None,
        normalize: bool = True,
    ) -> None:

        assert dataset_type in ["classification", "regression"], (
            f"Invalid dataset type provided. Got {dataset_type} but expected one of "
            f"['classification', 'regression']."
        )
        self.dataset_type: str = dataset_type
        self.random_state: Optional[int] = random_state

        # set the loss function
        self.loss_function: Callable[[np.ndarray, np.ndarray], float] = loss_function
        if self.loss_function is None:
            raise ValueError("No loss function provided.")
        self._empty_coalition_value: float = 0.0  # is set to 0 for all games

        # create the sanitized ensemble members list
        self.available_members: list[str] = [
            "regression",
            "decision_tree",
            "random_forest",
            "knn",
            "svm",
            "gradient_boosting",
        ]
        if ensemble_members is None:
            ensemble_members = []
            for i in range(n_members):
                ensemble_members.append(self.available_members[i % len(self.available_members)])
        else:  # check if all ensemble members are available
            for member in ensemble_members:
                if member not in self.available_members:
                    raise ValueError(
                        f"Invalid ensemble member provided. Got {member} but expected one of "
                        f"{self.available_members}."
                    )

        # setup base game and attributes
        self.player_names: list[str] = ensemble_members
        n_players: int = len(ensemble_members)
        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=self._empty_coalition_value,  # is set to 0 for all games
        )

        # init ensemble members
        self.ensemble_members: dict[int, Model] = self._init_ensemble_members()

        # fit the ensemble members
        for member_id, member in self.ensemble_members.items():
            if verbose:
                print(
                    f"Fitting ensemble member {member_id + 1} ({self.player_names[member_id]})  ..."
                )
            member.fit(x_train, y_train)

        # compute the predictions of the ensemble members
        self.predictions: np.ndarray = np.zeros((n_players, y_test.shape[0]))
        for member_id, member in self.ensemble_members.items():
            self.predictions[member_id] = member.predict(x_test)

        # store the test labels
        self._y_test: np.ndarray = y_test

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Computes the worth of the coalition for the ensemble selection game.

        The worth of a coalition is the performance of the ensemble on the test set as measured by
        a goodness_of_fit function.

        Args:
            coalitions: The coalitions as a binary matrix of shape (n_coalitions, n_players).

        Returns:
            The worth of the coalition.
        """
        worth = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            if sum(coalition) == 0:
                worth[i] = self._empty_coalition_value
                continue
            coalition_predictions = self.predictions[coalition].mean(axis=0)
            if self.dataset_type == "classification":
                coalition_predictions = np.round(coalition_predictions)  # round to class labels
            worth[i] = self.loss_function(self._y_test, coalition_predictions)
        return worth

    def _init_ensemble_members(self) -> dict[int, Model]:
        """Initializes the ensemble members."""
        ensemble_members: dict[int, Model] = {}
        for member_id, member in enumerate(self.player_names):
            if member == "regression":
                from sklearn.linear_model import LinearRegression, LogisticRegression

                if self.dataset_type == "classification":
                    model = LogisticRegression(random_state=self.random_state)
                else:
                    model = LinearRegression()
            elif member == "decision_tree":
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

                if self.dataset_type == "classification":
                    model = DecisionTreeClassifier(random_state=self.random_state)
                else:
                    model = DecisionTreeRegressor()
            elif member == "random_forest":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

                if self.dataset_type == "classification":
                    model = RandomForestClassifier(n_estimators=10, random_state=self.random_state)
                else:
                    model = RandomForestRegressor(n_estimators=10)
            elif member == "knn":
                from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

                if self.dataset_type == "classification":
                    model = KNeighborsClassifier(n_neighbors=3)
                else:
                    model = KNeighborsRegressor()
            elif member == "svm":
                from sklearn.svm import SVC, SVR

                if self.dataset_type == "classification":
                    model = SVC(random_state=self.random_state)
                else:
                    model = SVR()
            elif member == "gradient_boosting":
                from xgboost import XGBClassifier, XGBRegressor

                if self.dataset_type == "classification":
                    model = XGBClassifier(random_state=self.random_state)
                else:
                    model = XGBRegressor()
            else:
                raise ValueError(
                    f"Invalid ensemble member provided. Got {member} but expected one of "
                    f"{self.available_members}."
                )

            ensemble_members[member_id] = model
        return ensemble_members
