"""This module contains the Base Regression approximator to compute SII and k-SII of arbitrary max_order."""

from typing import Callable, Optional

import numpy as np
from scipy.special import binom, factorial

from shapiq.approximator._base import Approximator
from shapiq.approximator.k_sii import KShapleyMixin
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.interaction_values import InteractionValues

AVAILABLE_INDICES_REGRESSION = ["k-SII", "SII", "STII", "FSII"]


class MonteCarlo(Approximator, KShapleyMixin):
    """This class is the base class for all MonteCarlo approximators, e.g. SHAP-IQ and SVARM-IQ.

    MonteCarlo approximators are based on a representation of the interaction index as a weighted sum over discrete
    derivatives. The sum is re-written and approximated using Monte Carlo sampling.
    The sum may be stratified by coalition size or by the intersection size of the coalition and the interaction.
    The standard form for approximation is based on Theorem 1 in
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html

    Args:
        n: The number of players.
        max_order: The interaction order of the approximation.
        index: The interaction index to be estimated. Available indices are 'SII', 'kSII', 'STII',
            and 'FSII'.
        stratify_coalition_size: If True, then each coalition size is estimated separately
        stratify_intersection_size: If True, then each coalition is stratified by number of interseection elements
        top_order: If True, then only highest order interaction values are computed, e.g. required for FSII
        random_state: The random state to use for the approximation. Defaults to None.
    """

    def __init__(
        self,
        n: int,
        max_order: int,
        index: str,
        stratify_coalition_size: bool = True,
        stratify_intersection_size: bool = True,
        top_order: bool = False,
        random_state: Optional[int] = None,
    ):
        if index not in AVAILABLE_INDICES_REGRESSION:
            raise ValueError(
                f"Index {index} not available for Regression Approximator. Choose from "
                f"{AVAILABLE_INDICES_REGRESSION}."
            )
        super().__init__(
            n,
            min_order=0,
            max_order=max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
        )
        self._big_M: int = 10e7
        self.stratify_coalition_size = stratify_coalition_size
        self.stratify_intersection_size = stratify_intersection_size

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

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
        pairing_trick: bool = False,
        sampling_weights: np.ndarray = None,
    ) -> InteractionValues:
        if sampling_weights is None:
            # Initialize default sampling weights
            sampling_weights = self._init_sampling_weights()

        sampler = CoalitionSampler(
            n_players=self.n,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=self._random_state,
        )

        sampler.sample(budget)

        coalitions_matrix = sampler.coalitions_matrix
        coalitions_size = np.sum(coalitions_matrix, axis=1)
        coalitions_counter = sampler.coalitions_counter
        coalitions_size_probs = sampler.coalitions_size_probability
        coalitions_in_size_probs = sampler.coalitions_in_size_probability
        is_coalition_sampled = sampler.is_coalition_sampled
        sampling_size_probabilities = sampler.sampling_size_probabilities

        # query the game for the current batch of coalitions
        game_values = game(coalitions_matrix)

        if self.index == "k-SII":
            # For k-SII approximate SII values and then aggregate
            index_approximation = "SII"
        else:
            index_approximation = self.index

        shapley_interactions_values = self.montecarlo_routine(
            game_values,
            coalitions_matrix,
            coalitions_size,
            index_approximation,
            coalitions_counter,
            coalitions_size_probs,
            coalitions_in_size_probs,
            is_coalition_sampled,
            sampling_size_probabilities,
        )

        if np.shape(coalitions_matrix)[0] >= 2**self.n:
            estimated_indicator = False
        else:
            estimated_indicator = True

        if self.index == "k-SII":
            baseline_value = shapley_interactions_values[0]
            shapley_interactions_values = self.transforms_sii_to_ksii(shapley_interactions_values)
            if self.min_order == 0:
                shapley_interactions_values[0] = baseline_value

        return self._finalize_result(
            result=shapley_interactions_values, estimated=estimated_indicator, budget=budget
        )

    def montecarlo_routine(
        self,
        game_values: np.ndarray,
        coalitions_matrix: np.ndarray,
        coalitions_size: np.ndarray,
        index_approximation: str,
        coalitions_counter,
        coalitions_size_probs,
        coalitions_in_size_probs,
        is_coalition_sampled,
        sampling_size_probabilities,
    ):
        standard_form_weights = self._get_standard_form_weights(index_approximation)
        shapley_interaction_values = np.zeros(len(self.interaction_lookup))
        emptycoalition_value = game_values[coalitions_size == 0][0]
        game_values_centered = game_values - emptycoalition_value
        n_coalitions = len(game_values_centered)

        for interaction, interaction_pos in self.interaction_lookup.items():
            interaction_binary = np.zeros(self.n, dtype=int)
            interaction_binary[list(interaction)] = 1
            interaction_size = len(interaction)
            intersections_size = np.sum(coalitions_matrix * interaction_binary, axis=1)
            interaction_weights = standard_form_weights[
                interaction_size, coalitions_size, intersections_size
            ]

            # Default SHAP-IQ routine
            n_samples = np.sum(coalitions_counter[is_coalition_sampled])
            n_samples_helper = np.array([1, n_samples])
            # n_samples for sampled coalitions, else 1
            coalitions_n_samples = n_samples_helper[is_coalition_sampled.astype(int)]
            sampling_adjustment_weights = coalitions_counter / (
                coalitions_size_probs * coalitions_in_size_probs * coalitions_n_samples
            )

            if self.stratify_coalition_size and self.stratify_intersection_size:
                # Default SVARM-IQ Routine
                size_strata = np.unique(coalitions_size)
                intersection_strata = np.unique(intersections_size)
                for intersection_stratum in intersection_strata:
                    for size_stratum in size_strata:
                        in_stratum = (intersections_size == intersection_stratum) * (
                            coalitions_size == size_stratum
                        )
                        in_stratum_and_sampled = in_stratum * is_coalition_sampled
                        stratum_probabilities = np.ones(n_coalitions)
                        stratum_probabilities[in_stratum_and_sampled] = 1 / binom(
                            self.n - interaction_size,
                            coalitions_size[in_stratum_and_sampled] - intersection_stratum,
                        )
                        # Get sampled coalitions per stratum
                        stratum_n_samples = np.sum(coalitions_counter[in_stratum_and_sampled])
                        n_samples_helper = np.array([1, stratum_n_samples])
                        coalitions_n_samples = n_samples_helper[in_stratum_and_sampled.astype(int)]
                        sampling_adjustment_weights[in_stratum] = coalitions_counter[in_stratum] / (
                            coalitions_n_samples[in_stratum] * stratum_probabilities[in_stratum]
                        )

            elif self.stratify_coalition_size and not self.stratify_intersection_size:
                size_strata = np.unique(coalitions_size)
                for size_stratum in size_strata:
                    in_stratum = coalitions_size == size_stratum
                    in_stratum_and_sampled = in_stratum * is_coalition_sampled
                    stratum_probabilities = np.ones(n_coalitions)
                    stratum_probabilities[in_stratum_and_sampled] = 1 / binom(
                        self.n,
                        coalitions_size[in_stratum_and_sampled],
                    )
                    # Get sampled coalitions per stratum
                    stratum_n_samples = np.sum(coalitions_counter[in_stratum_and_sampled])
                    n_samples_helper = np.array([1, stratum_n_samples])
                    coalitions_n_samples = n_samples_helper[in_stratum_and_sampled.astype(int)]
                    sampling_adjustment_weights[in_stratum] = coalitions_counter[in_stratum] / (
                        coalitions_n_samples[in_stratum] * stratum_probabilities[in_stratum]
                    )

            elif not self.stratify_coalition_size and self.stratify_intersection_size:
                intersection_strata = np.unique(intersections_size)
                for intersection_stratum in intersection_strata:
                    # Flag all coalitions that belong to the stratum and are sampled
                    in_stratum = intersections_size == intersection_stratum
                    in_stratum_and_sampled = in_stratum * is_coalition_sampled
                    # Compute probabilities for a sample to be placed in this stratum
                    stratum_probabilities = np.ones(n_coalitions)
                    stratum_probabilities[in_stratum_and_sampled] = 1 / binom(
                       self.n - interaction_size,
                        coalitions_size[in_stratum_and_sampled] - intersection_stratum,
                    )
                    #stratum_probabilities = np.sum(sampling_size_probabilities*self.n)
                    # Get sampled coalitions per stratum
                    stratum_n_samples = np.sum(coalitions_counter[in_stratum_and_sampled])
                    n_samples_helper = np.array([1, stratum_n_samples])
                    coalitions_n_samples = n_samples_helper[in_stratum_and_sampled.astype(int)]

                    sampling_adjustment_weights[in_stratum] = coalitions_counter[in_stratum] / (
                        coalitions_n_samples[in_stratum]
                        * stratum_probabilities[in_stratum]
                        * coalitions_size_probs[in_stratum]
                    )

                shapley_interaction_values[interaction_pos] = np.sum(
                    game_values_centered * interaction_weights * sampling_adjustment_weights
                )
            else:
                # Default SHAP-IQ routine
                n_samples = np.sum(coalitions_counter[is_coalition_sampled])
                n_samples_helper = np.array([1, n_samples])
                # n_samples for sampled coalitions, else 1
                coalitions_n_samples = n_samples_helper[is_coalition_sampled.astype(int)]
                sampling_adjustment_weights = coalitions_counter / (
                    coalitions_size_probs * coalitions_in_size_probs * coalitions_n_samples
                )

            shapley_interaction_values[interaction_pos] = np.sum(
                game_values_centered * interaction_weights * sampling_adjustment_weights
            )

        if self.min_order == 0:
            # Set emptyset interaction manually to baseline, required for SII
            shapley_interaction_values[0] = emptycoalition_value

        return shapley_interaction_values

    def _sii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the SII discrete derivative weight given the coalition size and interaction size.

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        return 1 / (
            (self.n - interaction_size + 1) * binom(self.n - interaction_size, coalition_size)
        )

    def _stii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the STII discrete derivative weight given the coalition size and interaction size.

        Representation according to https://proceedings.mlr.press/v119/sundararajan20a.html

        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        if interaction_size == self.max_order:
            return self.max_order / (self.n * binom(self.n - 1, coalition_size))
        else:
            return 1.0 * (coalition_size == 0)

    def _fsii_weight(self, coalition_size: int, interaction_size: int) -> float:
        """Returns the FSII discrete derivative weight given the coalition size and interaction size.

        Representation according to Theorem 19 in https://www.jmlr.org/papers/volume24/22-0202/22-0202.pdf
        Args:
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        if interaction_size == self.max_order:
            return (
                factorial(2 * self.max_order - 1)
                / factorial(self.max_order - 1) ** 2
                * factorial(self.n - coalition_size - 1)
                * factorial(coalition_size + self.max_order - 1)
                / factorial(self.n + self.max_order - 1)
            )
        else:
            raise ValueError("Lower order interactions are not supported.")

    def _weight(self, index: str, coalition_size: int, interaction_size: int) -> float:
        """Returns the weight for each interaction type given coalition and interaction size.

        Args:
            index: The interaction index
            coalition_size: The size of the subset.
            interaction_size: The size of the interaction.

        Returns:
            float: The weight for the interaction type.
        """
        if index == "SII":  # SII default
            return self._sii_weight(coalition_size, interaction_size)
        elif index == "STII":
            return self._stii_weight(coalition_size, interaction_size)
        elif index == "FSII":
            return self._fsii_weight(coalition_size, interaction_size)
        else:
            raise ValueError(f"Unknown index {index}.")

    def _get_standard_form_weights(self, index: str) -> dict[int, np.ndarray[float]]:
        """Initializes the weights for the interaction index re-written from discrete derivatives to standard form.
         Standard form according to Theorem 1 in https://proceedings.neurips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html

        Args:
            index: The interaction index

        Returns:
            The standard form weights.
        """
        # init data structure
        weights = np.zeros((self.max_order + 1, self.n + 1, self.max_order + 1))
        for order in self._order_iterator:
            # fill with values specific to each index
            for coalition_size in range(0, self.n + 1):
                for intersection_size in range(
                    max(0, order + coalition_size - self.n), min(order, coalition_size) + 1
                ):
                    weights[order, coalition_size, intersection_size] = (-1) ** (
                        order - intersection_size
                    ) * self._weight(index, coalition_size - intersection_size, order)
        return weights
