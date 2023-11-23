"""This module contains the base approximator classes for the shapiq package."""
import copy
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union, Optional

import numpy as np
from scipy.special import binom

AVAILABLE_INDICES = {"SII", "nSII", "STI", "FSI"}


@dataclass
class InteractionValues:
    """This class contains the interaction values as estimated by an approximator.

    Attributes:
        values: The interaction values of the model. Mapping from order to the interaction values.
        index: The interaction index estimated. Available indices are 'SII', 'nSII', 'STI', and
            'FSI'.
        order: The order of the approximation.
        estimated: Whether the interaction values are estimated or not. Defaults to True.
        estimation_budget: The budget used for the estimation. Defaults to None.
    """

    values: dict[int, np.ndarray]
    index: str
    order: int
    estimated: bool = True
    estimation_budget: Optional[int] = None

    def __post_init__(self) -> None:
        """Checks if the index is valid."""
        if self.index not in ["SII", "nSII", "STI", "FSI"]:
            raise ValueError(
                f"Index {self.index} is not valid. "
                f"Available indices are 'SII', 'nSII', 'STI', and 'FSI'."
            )
        if self.order < 1 or self.order != max(self.values.keys()):
            raise ValueError(
                f"Order {self.order} is not valid. "
                f"Order should be a positive integer equal to the maximum key of the values."
            )

    def __repr__(self) -> str:
        """Returns the representation of the InteractionValues object."""
        representation = f"InteractionValues(\n"
        representation += (
            f"    index={self.index}, order={self.order}, estimated={self.estimated}"
            f", estimation_budget={self.estimation_budget},\n"
        ) + "    values={"
        for order, values in self.values.items():
            representation += "\n"
            representation += f"        {order}: "
            string_values: str = str(np.round(values, 4))
            string_values = string_values.replace("-0. ", " 0. ")
            string_values = string_values.replace("-0.]", " 0.]")
            string_values = string_values.replace("[-0. ", "[ 0. ")
            representation += string_values.replace("\n", "\n" + " " * 11)
        representation += "\n    }"
        representation += "\n)"
        return representation

    def __str__(self) -> str:
        """Returns the string representation of the InteractionValues object."""
        return self.__repr__()

    def __getitem__(self, item: int) -> np.ndarray:
        """Returns the interaction values for the given order.

        Args:
            item: The order of the interaction values.

        Returns:
            The interaction values.
        """
        return self.values[item]

    def __eq__(self, other: object) -> bool:
        """Checks if two InteractionValues objects are equal.

        Args:
            other: The other InteractionValues object.

        Returns:
            True if the two objects are equal, False otherwise.
        """
        if not isinstance(other, InteractionValues):
            raise NotImplementedError("Cannot compare InteractionValues with other types.")
        if self.index != other.index or self.order != other.order:
            return False
        for order, values in self.values.items():
            if not np.allclose(values, other.values[order]):
                return False
        return True

    def __ne__(self, other: object) -> bool:
        """Checks if two InteractionValues objects are not equal.

        Args:
            other: The other InteractionValues object.

        Returns:
            True if the two objects are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Returns the hash of the InteractionValues object."""
        return hash((self.index, self.order, tuple(self.values.values())))

    def __copy__(self) -> "InteractionValues":
        """Returns a copy of the InteractionValues object."""
        return InteractionValues(
            values=copy.deepcopy(self.values),
            index=self.index,
            order=self.order,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
        )

    def __deepcopy__(self, memo) -> "InteractionValues":
        """Returns a deep copy of the InteractionValues object."""
        return InteractionValues(
            values=copy.deepcopy(self.values),
            index=self.index,
            order=self.order,
            estimated=self.estimated,
            estimation_budget=self.estimation_budget,
        )


class Approximator(ABC):
    """This class is the base class for all approximators.

    Approximators are used to estimate the interaction values of a model or any value function.
    Different approximators can be used to estimate different interaction indices. Some can be used
    to estimate all indices.

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated. Available indices are 'SII', 'nSII', 'STI',
            and 'FSI'.
        top_order: If True, the approximation is performed only for the top order interactions. If
            False, the approximation is performed for all orders up to the specified order.
        random_state: The random state to use for the approximation. Defaults to None.

    Attributes:
        n: The number of players.
        N: The set of players (starting from 0 to n - 1).
        N_arr: The array of players (starting from 0 to n).
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated.
        top_order: If True, the approximation is performed only for the top order interactions. If
            False, the approximation is performed for all orders up to the specified order.
        min_order: The minimum order of the approximation. If top_order is True, min_order is equal
            to max_order. Otherwise, min_order is equal to 1.

    Properties:
        iteration_cost: The cost of a single iteration of the approximation.
    """

    @abstractmethod
    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        top_order: bool,
        random_state: Optional[int] = None,
    ) -> None:
        """Initializes the approximator."""
        self.index: str = index
        if self.index not in AVAILABLE_INDICES:
            raise ValueError(
                f"Index {self.index} is not valid. " f"Available indices are {AVAILABLE_INDICES}."
            )
        self.n: int = n
        self.N: set = set(range(self.n))
        self.N_arr: np.ndarray[int] = np.arange(self.n + 1)
        self.max_order: int = max_order
        self.top_order: bool = top_order
        self.min_order: int = self.max_order if self.top_order else 1
        self._random_state: Optional[int] = random_state
        self._rng: Optional[np.random.Generator] = np.random.default_rng(seed=self._random_state)
        self._iteration_cost: Optional[int] = None

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

    @property
    def iteration_cost(self) -> int:
        """Returns the cost of a single iteration of the approximation.

        Returns:
            The cost of a single iteration.
        """
        if self._iteration_cost is None:
            if hasattr(self, "_compute_iteration_cost"):
                self._iteration_cost = self._compute_iteration_cost()
            else:
                return 1
        return self._iteration_cost

    def _init_result(self, dtype=float) -> dict[int, np.ndarray]:
        """Initializes the result dictionary mapping from order to the interaction values.
        For order 1 the interaction values are of shape (n,) for order 2 of shape (n, n) and so on.

        Args:
            dtype: The data type of the result dictionary values. Defaults to float.

        Returns:
            The result dictionary.
        """
        result = {s: self._get_empty_array(self.n, s, dtype=dtype) for s in self._order_iterator}
        return result

    @staticmethod
    def _get_empty_array(n: int, order: int, dtype=float) -> np.ndarray:
        """Returns an empty array of the appropriate shape for the given order.

        Args:
            n: The number of players.
            order: The order of the array.
            dtype: The data type of the array. Defaults to float.

        Returns:
            The empty array.
        """
        return np.zeros(n**order, dtype=dtype).reshape((n,) * order)

    @property
    def _order_iterator(self) -> range:
        """Returns an iterator over the orders of the approximation.

        Returns:
            The iterator.
        """
        return range(self.min_order, self.max_order + 1)

    def _finalize_result(
        self, result, estimated: bool = True, budget: Optional[int] = None
    ) -> InteractionValues:
        """Finalizes the result dictionary.

        Args:
            result: The result dictionary.
            estimated: Whether the interaction values are estimated or not. Defaults to True.
            budget: The budget used for the estimation. Defaults to None.

        Returns:
            The interaction values.
        """
        return InteractionValues(
            values=result,
            index=self.index,
            order=self.max_order,
            estimated=estimated,
            estimation_budget=budget,
        )

    @staticmethod
    def _smooth_with_epsilon(
        interaction_results: Union[dict, np.ndarray], eps=0.00001
    ) -> Union[dict, np.ndarray]:
        """Smooth the interaction results with a small epsilon to avoid numerical issues.

        Args:
            interaction_results: Interaction results.
            eps: Small epsilon. Defaults to 0.00001.

        Returns:
            Union[dict, np.ndarray]: Smoothed interaction results.
        """
        if not isinstance(interaction_results, dict):
            interaction_results[np.abs(interaction_results) < eps] = 0
            return copy.deepcopy(interaction_results)
        interactions = {}
        for interaction_order, interaction_values in interaction_results.items():
            interaction_values[np.abs(interaction_values) < eps] = 0
            interactions[interaction_order] = interaction_values
        return copy.deepcopy(interactions)

    @staticmethod
    def _get_n_iterations(budget: int, batch_size: int, iteration_cost: int) -> tuple[int, int]:
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

    @staticmethod
    def _get_explicit_subsets(n: int, subset_sizes: list[int]) -> np.ndarray[bool]:
        """Enumerates all subsets of the given sizes and returns a one-hot matrix.

        Args:
            n: number of players.
            subset_sizes: list of subset sizes.

        Returns:
            one-hot matrix of all subsets of certain sizes.
        """
        total_subsets = int(sum(binom(n, size) for size in subset_sizes))
        subset_matrix = np.zeros(shape=(total_subsets, n), dtype=bool)
        subset_index = 0
        for subset_size in subset_sizes:
            for subset in itertools.combinations(range(n), subset_size):
                subset_matrix[subset_index, subset] = True
                subset_index += 1
        return subset_matrix

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
        """Checks if two Approximator objects are equal.

        Args:
            other: The other Approximator object.

        Returns:
            True if the two objects are equal, False otherwise.
        """
        if not isinstance(other, Approximator):
            raise NotImplementedError("Cannot compare Approximator with other types.")
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
        """Checks if two Approximator objects are not equal.

        Args:
            other: The other Approximator object.

        Returns:
            True if the two objects are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Returns the hash of the Approximator object."""
        return hash((self.n, self.max_order, self.index, self.top_order, self._random_state))

    def __copy__(self) -> "Approximator":
        """Returns a copy of the Approximator object."""
        return self.__class__(
            n=self.n,
            max_order=self.max_order,
            index=self.index,
            top_order=self.top_order,
            random_state=self._random_state,
        )

    def __deepcopy__(self, memo) -> "Approximator":
        """Returns a deep copy of the Approximator object."""
        return self.__class__(
            n=self.n,
            max_order=self.max_order,
            index=self.index,
            top_order=self.top_order,
            random_state=self._random_state,
        )


class ShapleySamplingMixin:
    """Mixin class for the computation of Shapley weights.

    Provides the common functionality for regression-based approximators like
    :class:`~shapiq.approximators.RegressionFSI`. The class offers computation of Shapley weights
    and the corresponding sampling weights for the KernelSHAP-like estimation approaches. The mixin
    assumes that the subclasses have a parameter `n` that defines the number of players.
    """

    def __init__(self) -> None:
        """Initializes the regression mixin."""
        self._big_M = float(1_000_000)
        # check weather self is a subclass of Approximator
        if not issubclass(self.__class__, Approximator):
            raise TypeError("ShapleyWeightsMixin can only be used with subclasses of Approximator.")

    def _init_ksh_sampling_weights(self) -> np.ndarray[float]:
        """Initializes the weights for sampling subsets.

        The sampling weights are of size n + 1 and indexed by the size of the subset. The edges
        (the first, empty coalition, and the last element, full coalition) are set to 0.

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).
        """
        weight_vector = np.zeros(shape=self.n - 1, dtype=float)
        for subset_size in range(1, self.n):
            weight_vector[subset_size - 1] = (self.n - 1) / (subset_size * (self.n - subset_size))
        sampling_weight = (np.asarray([0] + [*weight_vector] + [0])) / sum(weight_vector)
        return sampling_weight

    def _get_ksh_subset_weights(self, subsets: np.ndarray[bool]) -> np.ndarray[float]:
        """Computes the KernelSHAP regression weights for the given subsets.

        The weights for the subsets of size s are set to ksh_weights[s] / binom(n, s). The weights
        for the empty and full sets are set to a big number.

        Args:
            subsets: one-hot matrix of subsets for which to compute the weights in shape
                (n_subsets, n).

        Returns:
            The KernelSHAP regression weights in shape (n_subsets,).
        """
        # set the weights for each subset to ksh_weights[|S|] / binom(n, |S|)
        ksh_weights = self._init_ksh_sampling_weights()  # indexed by subset size
        subset_sizes = np.sum(subsets, axis=1)
        weights = ksh_weights[subset_sizes]  # set the weights for each subset size
        weights /= binom(self.n, subset_sizes)  # divide by the number of subsets of the same size

        # set the weights for the empty and full sets to big M
        weights[np.logical_not(subsets).all(axis=1)] = self._big_M
        weights[subsets.all(axis=1)] = self._big_M
        return weights

    def _sample_subsets(
        self,
        budget: int,
        sampling_weights: np.ndarray[float],
        replacement: bool = False,
        pairing: bool = True,
    ) -> np.ndarray[bool]:
        """Samples subsets with the given budget.

        Args:
            budget: budget for the sampling.
            sampling_weights: weights for sampling subsets of certain sizes and indexed by the size.
                The shape is expected to be (n + 1,). A size that is not to be sampled has weight 0.
            pairing: whether to use pairing (`True`) sampling or not (`False`). Defaults to `False`.

        Returns:
            sampled subsets.
        """
        # sanitize input parameters
        sampling_weights = copy.copy(sampling_weights)
        sampling_weights /= np.sum(sampling_weights)

        # adjust budget for paired sampling
        if pairing:
            budget = budget - budget % 2  # must be even for pairing
            budget = int(budget / 2)

        # create storage array for given budget
        subset_matrix = np.zeros(shape=(budget, self.n), dtype=bool)

        # sample subsets
        sampled_sizes = self._rng.choice(self.N_arr, size=budget, p=sampling_weights).astype(int)
        if replacement:  # sample subsets with replacement
            permutations = np.tile(np.arange(self.n), (budget, 1))
            self._rng.permuted(permutations, axis=1, out=permutations)
            for i, subset_size in enumerate(sampled_sizes):
                subset = permutations[i, :subset_size]
                subset_matrix[i, subset] = True
        else:  # sample subsets without replacement
            sampled_subsets, n_sampled = set(), 0  # init sampling variables
            while n_sampled < budget:
                subset_size = sampled_sizes[n_sampled]
                subset = tuple(sorted(self._rng.choice(np.arange(0, self.n), size=subset_size)))
                sampled_subsets.add(subset)
                if len(sampled_subsets) != n_sampled:  # subset was not already sampled
                    subset_matrix[n_sampled, subset] = True
                    n_sampled += 1  # continue sampling

        if pairing:
            subset_matrix = np.repeat(subset_matrix, repeats=2, axis=0)  # extend the subset matrix
            subset_matrix[1::2] = np.logical_not(subset_matrix[1::2])  # flip sign of paired subsets

        return subset_matrix
