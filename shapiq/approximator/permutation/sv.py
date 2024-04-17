"""This module contains the permutation sampling approximation method for the Shapley value (SV): ApproShapley (Castro et al., 2009).
It estimates the Shapley values by sampling random permutations of the player set
and extracting all marginal contributions from each permutation."""

from typing import Callable, Optional

import numpy as np

from shapiq.approximator._base import Approximator
from shapiq.interaction_values import InteractionValues


class PermutationSamplingSV(Approximator):
    """The  Permutation Sampling algorithm ApproShapley estimates the Shapley values (SV) by sampling random permutations
    of the player set and extracting all marginal contributions from each permutation.
    For more information, see [Castro et al. (2009)](https://doi.org/10.1016/j.cor.2008.04.004).

    Args:
        n: The number of players.
        random_state: The random state to use for the permutation sampling. Defaults to `None`.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        N_arr: The array of players (starting from 0 to n).
        iteration_cost: The cost of a single iteration of the approximator.

    Examples:
        >>> from shapiq.approximator import PermutationSamplingSV
        >>> from shapiq.games import DummyGame
        >>> game = DummyGame(5, (1, 2))
        >>> approximator = PermutationSamplingSV(game.n_players, random_state=42)
        >>> sv_estimates = approximator.approximate(100, game)
        >>> print(sv_estimates.values)
        [0.2 0.7 0.7 0.2 0.2]
    """

    def __init__(
        self,
        n: int,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(n=n, max_order=1, index="SV", top_order=False, random_state=random_state)
        self.iteration_cost: int = n - 1

    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray], batch_size: Optional[int] = 5
    ) -> InteractionValues:
        """Approximates the Shapley values using ApproShapley.

        Args:
            budget: The number of game evaluations for approximation
            game: The game function as a callable that takes a set of players and returns the value.
            batch_size: The size of the batch. If None, the batch size is set to 1. Defaults to 5.

        Returns:
            The estimated interaction values.
        """

        result: np.ndarray[float] = self._init_result()
        counts: np.ndarray[int] = self._init_result(dtype=int)

        batch_size = 1 if batch_size is None else batch_size

        # store the values of the empty and full coalition
        # this saves 2 evaluations per permutation
        empty_val = game(np.zeros(self.n, dtype=bool))
        full_val = game(np.ones(self.n, dtype=bool))
        used_budget = 2

        # catch special case of single player game, otherwise iteration through permutations fails
        if self.n == 1:
            result[0] = full_val - empty_val
            counts[0] = 1
            return self._finalize_result(result, budget=used_budget, estimated=True)

        # compute the number of iterations and size of the last batch (can be smaller than original)
        n_iterations, last_batch_size = self._calc_iteration_count(
            budget - 2, batch_size, self.iteration_cost
        )

        # main permutation sampling loop
        for iteration in range(1, n_iterations + 1):
            batch_size = batch_size if iteration != n_iterations else last_batch_size

            # create batch_size many permutations: two-dimensional matrix of shape (batch_size, n)
            # each row is a permutation
            permutations = np.tile(np.arange(self.n), (batch_size, 1))
            self._rng.permuted(permutations, axis=1, out=permutations)
            n_permutations = batch_size
            n_coalitions = n_permutations * self.iteration_cost

            # generate all coalitions to evaluate from permutations
            coalitions = np.zeros(shape=(n_coalitions, self.n), dtype=bool)
            coalition_index = 0
            for permutation_id in range(n_permutations):
                permutation = permutations[permutation_id]
                coalition = set()
                for i in range(self.n - 1):
                    coalition.add(permutation[i])
                    coalitions[coalition_index, tuple(coalition)] = True
                    coalition_index += 1

            # evaluate the collected coalitions
            game_values: np.ndarray[float] = game(coalitions)
            used_budget += len(coalitions)

            # update the estimates
            coalition_index = 0
            for permutation_id in range(n_permutations):
                # update the first player in the permutation
                permutation = permutations[permutation_id]
                marginal_con = game_values[coalition_index] - empty_val
                result[permutation[0]] += marginal_con
                counts[permutation[0]] += 1
                # update the players in the middle of the permutation
                for i in range(1, self.n - 1):
                    marginal_con = game_values[coalition_index + 1] - game_values[coalition_index]
                    result[permutation[i]] += marginal_con
                    counts[permutation[i]] += 1
                    coalition_index += 1
                # update the last player in the permutation
                marginal_con = full_val - game_values[coalition_index]
                result[permutation[self.n - 1]] += marginal_con
                counts[permutation[self.n - 1]] += 1
                coalition_index += 1

        result = np.divide(result, counts, out=result, where=counts != 0)
        return self._finalize_result(result, budget=used_budget, estimated=True)
