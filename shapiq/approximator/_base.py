"""This module contains the base approximator classes for the shapiq package."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from shapiq.approximator._config import AVAILABLE_INDICES
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import generate_interaction_lookup

__all__ = [
    "Approximator",
]


class Approximator(ABC):
    """This class is the base class for all approximators.

    Approximators are used to estimate the interaction values of a model or any value function.
    Different approximators can be used to estimate different interaction indices. Some can be used
    to estimate all indices.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        min_order: The minimum interaction order, default is 1, but regression uses 0 currently TODO:rework
        index: The interaction index to be estimated. Available indices are 'SII', 'kSII', 'STII',
            and 'FSII'.
        top_order: If True, the approximation is performed only for the top order interactions. If
            False, the approximation is performed for all orders up to the specified order.
        pairing_trick: If `True`, the pairing trick is applied to the sampling procedure. Defaults
            to `False`.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape `(n + 1,)` and are used to determine the probability of sampling a coalition
             of a certain size. Defaults to `None`.
        random_state: The random state to use for the approximation. Defaults to None.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1). TODO: Change to _grand_coalition_set
        N_arr: The array of players (starting from 0 to n).
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated.
        top_order: If True, the approximation is performed only for the top order interactions. If
            False, the approximation is performed for all orders up to the specified order.
        min_order: The minimum order of the approximation. If top_order is True, min_order is equal
            to max_order. Otherwise, min_order is equal to 0.
        iteration_cost: The cost of a single iteration of the approximator.

    """

    @abstractmethod
    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        top_order: bool,
        min_order: int = 1,
        random_state: Optional[int] = None,
        pairing_trick: bool = False,
        sampling_weights: Optional[np.ndarray[float]] = None,
    ) -> None:
        """Initializes the approximator."""
        self.index: str = index
        if self.index not in AVAILABLE_INDICES:
            raise ValueError(
                f"Index {self.index} is not valid. " f"Available indices are {AVAILABLE_INDICES}."
            )
        self.n: int = n
        self.N: set = set(range(self.n))  # TODO: remove from approximator
        self._grand_coalition_set = set(range(self.n))
        self.N_arr: np.ndarray[int] = np.arange(self.n + 1, dtype=int)
        self.top_order: bool = top_order
        self.max_order: int = max_order
        self.min_order: int = self.max_order if self.top_order else min_order
        self.iteration_cost: int = 1  # default value, can be overwritten by subclasses
        self._interaction_lookup = generate_interaction_lookup(
            self.n, self.min_order, self.max_order
        )
        self._random_state: Optional[int] = random_state
        self._rng: Optional[np.random.Generator] = np.random.default_rng(seed=self._random_state)

        # set up the coalition sampler
        self._big_M: int = 100_000_000  # large number for sampling weights
        if sampling_weights is None:  # init default sampling weights
            sampling_weights = self._init_sampling_weights()
        self._sampler = CoalitionSampler(
            n_players=self.n,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=self._random_state,
        )

    @abstractmethod
    def approximate(
        self, budget: int, game: Callable[[np.ndarray], np.ndarray], *args, **kwargs
    ) -> InteractionValues:
        """Approximates the interaction values. Abstract method that needs to be implemented for
        each approximator.

        Args:
            budget: The budget for the approximation.
            game: The game function.

        Returns:
            The interaction values.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("The approximate method needs to be implemented.")

    def _init_sampling_weights(self) -> np.ndarray:
        """Initializes the weights for sampling subsets.

        The sampling weights are of size n + 1 and indexed by the size of the subset. The edges
        All weights are set to _big_M, if size < order or size > n - order to ensure efficiency.

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).
        """
        weight_vector = np.zeros(shape=self.n + 1)
        for coalition_size in range(0, self.n + 1):
            if (coalition_size == 0) or (coalition_size == self.n):
                # prioritize these subsets
                weight_vector[coalition_size] = self._big_M**2
            elif (coalition_size < self.max_order) or (coalition_size > self.n - self.max_order):
                # prioritize these subsets
                weight_vector[coalition_size] = self._big_M
            else:
                # KernelSHAP sampling weights
                weight_vector[coalition_size] = 1 / (coalition_size * (self.n - coalition_size))
        sampling_weight = weight_vector / np.sum(weight_vector)
        return sampling_weight

    def _init_result(self, dtype=float) -> np.ndarray:
        """Initializes the result array. The result array is a 1D array of size n_interactions as
        determined by the interaction_lookup dictionary.

        Args:
            dtype: The data type of the result array. Defaults to float.

        Returns:
            The result array.
        """
        result = np.zeros(len(self._interaction_lookup), dtype=dtype)
        return result

    @property
    def _order_iterator(self) -> range:
        """Returns an iterator over the orders of the approximation.

        Returns:
            The iterator.
        """
        return range(self.min_order, self.max_order + 1)

    def _finalize_result(
        self,
        result,
        estimated: bool = True,
        budget: Optional[int] = None,
        index: Optional[str] = None,
    ) -> InteractionValues:
        """Finalizes the result dictionary.

        Args:
            result: The result dictionary.
            estimated: Whether the interaction values are estimated or not. Defaults to True.
            budget: The budget used for the estimation. Defaults to None.
            index: The interaction index estimated. Available indices are 'SII', 'kSII', 'STII', and
                'FSII'. Defaults to None (i.e., the index of the approximator is used).

        Returns:
            The interaction values.
        """
        if tuple() in self.interaction_lookup:
            baseline_value = result[
                0
            ]  # Set baseline value in for emptyset in values, if available in lookup
        else:
            baseline_value = 0.0

        if index is None:
            index = self.index
        return InteractionValues(
            values=result,
            estimated=estimated,
            estimation_budget=budget,
            index=index,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            interaction_lookup=self._interaction_lookup,
            baseline_value=baseline_value,
        )

    @staticmethod
    def _calc_iteration_count(budget: int, batch_size: int, iteration_cost: int) -> tuple[int, int]:
        """Computes the number of iterations and the size of the last batch given the batch size and
        the budget.

        Args:
            budget: The budget for the approximation.
            batch_size: The size of the batch.
            iteration_cost: The cost of a single iteration.

        Returns:
            int, int: The number of iterations and the size of the last batch.
        """
        n_iterations = budget // (iteration_cost * batch_size)
        last_batch_size = batch_size
        remaining_budget = budget - n_iterations * iteration_cost * batch_size
        if remaining_budget > 0 and remaining_budget // iteration_cost > 0:
            last_batch_size = remaining_budget // iteration_cost
            n_iterations += 1
        return n_iterations, last_batch_size

    def __repr__(self) -> str:
        """Returns the representation of the Approximator object."""
        return (
            f"{self.__class__.__name__}(\n"
            f"    n={self.n},\n"
            f"    max_order={self.max_order},\n"
            f"    index={self.index},\n"
            f"    top_order={self.top_order},\n"
            f"    random_state={self._random_state}\n"
            f")"
        )

    def __str__(self) -> str:
        """Returns the string representation of the Approximator object."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Checks if two Approximator objects are equal."""
        if not isinstance(other, Approximator):
            raise ValueError("Cannot compare Approximator with other types.")
        if (
            self.n != other.n
            or self.max_order != other.max_order
            or self.index != other.index
            or self.top_order != other.top_order
            or self._random_state != other._random_state
        ):
            return False
        return True

    def __ne__(self, other: object) -> bool:
        """Checks if two Approximator objects are not equal."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Returns the hash of the Approximator object."""
        return hash((self.n, self.max_order, self.index, self.top_order, self._random_state))

    @property
    def interaction_lookup(self):
        return self._interaction_lookup
