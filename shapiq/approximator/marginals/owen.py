"""This module contains the Owen Sampling approximation method for the Shapley value (SV) by Okhrati and Lipani (2020).
It estimates the Shapley values in its integral representation by sampling random marginal contributions."""

from typing import Callable, Optional

import numpy as np

from shapiq.approximator._base import Approximator
from shapiq.interaction_values import InteractionValues

AVAILABLE_INDICES_SHAPIQ = {"SV"}


class OwenSamplingSV(Approximator):
    """The Owen Sampling algorithm estimates the Shapley values (SV) by sampling random marginal contributions
    for each player and each coalition size. The marginal contributions are used to update an integral representation of the SV.
    For more information, see [Okhrati and Lipani (2020)](https://www.computer.org/csdl/proceedings-article/icpr/2021/09412511/1tmicWxYo2Q).
    The number of anchor points M at which the integral is to be palpated share the avilable budget for each player equally.
    A higher M increases the resolution of the integral reducing bias while reducing the accuracy of the estimation at each point.

    Args:
        n: The number of players.
        random_state: The random state to use for the permutation sampling. Defaults to `None`.
        M: Number of anchor points at which the integral is to be palpated.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        N_arr: The array of players (starting from 0 to n).
        iteration_cost: The cost of a single iteration of the approximator.

    Examples:
        >>> from shapiq.approximator import StratifiedSamplingSV
        >>> from shapiq.games import DummyGame
        >>> game = DummyGame(5, (1, 2))
        >>> approximator = OwenSamplingSV(game.n_players, 10, random_state=42)
        >>> sv_estimates = approximator.approximate(100, game)
        >>> print(sv_estimates.values)
        [0.2 0.7 0.7 0.2 0.2]
    """

    def __init__(
        self,
        n: int,
        M: int,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(n, max_order=1, index="SV", top_order=False, random_state=random_state)
        self.iteration_cost: int = 2 * n * M
        self.M = M

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray], batch_size: Optional[int] = 5
    ) -> InteractionValues:
        """Approximates the Shapley values using Owen Sampling.

        Args:
            budget: The number of game evaluations for approximation
            game: The game function as a callable that takes a set of players and returns the value.
            batch_size: The size of the batch. If None, the batch size is set to 1. Defaults to 5.

        Returns:
            The estimated interaction values.
        """

        used_budget = 0
        batch_size = 1 if batch_size is None else batch_size

        # compute the number of iterations and size of the last batch (can be smaller than original)
        n_iterations, last_batch_size = self._calc_iteration_count(
            budget, batch_size, self.iteration_cost
        )

        anchors = self.get_anchor_points(self.M)
        estimates = np.zeros((self.n, self.M), dtype=float)
        counts = np.zeros((self.n, self.M), dtype=int)

        # main sampling loop going through all anchor points of all players with each segment
        for iteration in range(1, n_iterations + 1):
            batch_size = batch_size if iteration != n_iterations else last_batch_size
            n_segments = batch_size
            n_coalitions = n_segments * self.iteration_cost
            coalitions = np.zeros(shape=(n_coalitions, self.n), dtype=bool)
            coalition_index = 0
            # iterate through each segment
            for segment in range(n_segments):
                # iterate through each player
                for player in range(self.n):
                    # iterate through each anchor point
                    for q in anchors:
                        # draw a subset of players without player: all are inserted independently with probability q
                        coalition = np.random.choice(
                            [True, False], self.n - 1, replace=True, p=[q, 1 - q]
                        )
                        # add information that player is absent
                        coalition = np.insert(coalition, player, False)
                        coalitions[coalition_index] = coalition
                        # add information that player is present to complete marginal contribution
                        coalition[player] = True
                        coalitions[coalition_index + 1] = coalition
                        coalition_index += 2

            # evaluate the collected coalitions
            game_values: np.ndarray[float] = game(coalitions)
            used_budget += len(coalitions)

            # update the anchor estimates
            coalition_index = 0
            # iterate through each segment
            for segment in range(n_segments):
                for player in range(self.n):
                    for m in range(self.M):
                        # calculate the marginal contribution and update the anchor estimate
                        marginal_con = (
                            game_values[coalition_index + 1] - game_values[coalition_index]
                        )
                        estimates[player][m] += marginal_con
                        counts[player][m] += 1
                        coalition_index += 2

        # aggregate the anchor estimates: divide each anchor sum by its sample number, sum up the means, divide by the number of valid anchor estimates
        estimates = np.divide(estimates, counts, out=estimates, where=counts != 0)
        result = np.sum(estimates, axis=1)
        non_zeros = np.count_nonzero(counts, axis=1)
        result = np.divide(result, non_zeros, out=result, where=non_zeros != 0)
        return self._finalize_result(result, budget=used_budget, estimated=True)

    def get_anchor_points(self, m: int):
        if m <= 0:
            raise ValueError("The number of anchor points needs to be greater than 0.")

        if m == 1:
            return np.array([0.5])
        else:
            return np.linspace(0.0, 1.0, num=m)
