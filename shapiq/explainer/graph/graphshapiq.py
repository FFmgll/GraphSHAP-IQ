from typing import Union, Callable

from shapiq import powerset
import numpy as np
import copy

from scipy.special import binom

from shapiq.approximator.moebius_converter import MoebiusConverter
from shapiq import InteractionValues
from shapiq.games.benchmark.local_xai.benchmark_graph import GraphGame


class GraphSHAP:
    def __init__(self, n_players: int, game: GraphGame, edge_index: np.ndarray):
        self.n_players = n_players
        self._grand_coalition_set = set(range(n_players))
        self.game: GraphGame = game
        self.edge_index = copy.copy(edge_index)
        self._grand_coalition_prediction = game(np.ones(n_players))
        self._last_n_model_calls: int = None
    def _get_k_neighborhood(self, node, k):
        neighbors = set()
        queue = [(node, 0)]
        visited = set([node])
        while queue:
            curr_node, dist = queue.pop(0)
            if dist > k:
                break
            if dist <= k:
                neighbors.add(curr_node)
            if dist < k:
                # Find neighbors of current node
                for edge in self.edge_index.T:
                    if edge[0] == curr_node and edge[1] not in visited:
                        queue.append((edge[1], dist + 1))
                        visited.add(edge[1])
        return tuple(sorted(neighbors))

    def compute_moebius_transform(
        self,
        coalitions: dict,
        coalition_predictions: np.ndarray,
        coalition_lookup: np.ndarray,
    ):
        moebius_values = np.zeros(len(coalitions))
        moebius_lookup = {}

        for i, coalition in enumerate(coalitions):
            moebius_values[i] = 0
            moebius_lookup[coalition] = i
            for L in powerset(coalition):
                moebius_values[i] += (-1) ** (len(coalition) - len(L)) * coalition_predictions[
                    coalition_lookup[L]
                ]

        moebius_coefficients = InteractionValues(
            values=moebius_values,
            interaction_lookup=moebius_lookup,
            min_order=0,
            max_order=self.n_players,
            n_players=self.n_players,
            index="Moebius",
        )

        return moebius_coefficients

    def _convert_to_coalition_matrix(self, coalitions: Union[set, dict], lookup_shift: int = 0):
        coalition_matrix = np.zeros((len(coalitions), self.n_players))
        coalition_lookup = {}

        if type(coalitions) == set:
            for i, S in enumerate(coalitions):
                coalition_matrix[i, S] = 1
                coalition_lookup[S] = lookup_shift + i
        if type(coalitions) == dict:
            for i, (node, S) in enumerate(coalitions.items()):
                coalition_matrix[i, S] = 1
                coalition_lookup[S] = lookup_shift + i
        return coalition_matrix, coalition_lookup

    def explain(
        self,
        max_neighborhood_size: int,
        max_interaction_size: int,
        order: int,
        efficiency_routine: bool = True,
    ):

        # Get non-zero Möbius values based on the neighborhood
        neighbors = {}
        incomplete_neighborhoods = set()
        moebius_interactions = set()
        for node in self._grand_coalition_set:
            # Compute k-neighborhood of each node with maximum range max_neighborhood_size
            neighbors[node] = self._get_k_neighborhood(node, max_neighborhood_size)
            # Compute maximum size of interactions in the neighborhood
            max_interaction_size = min(len(neighbors[node]), max_interaction_size)
            # Collect all non-zero Möbius interactions up to order max_interaction_size
            # For these, game evaluations are required
            for interaction in powerset(neighbors[node], max_size=max_interaction_size):
                moebius_interactions.add(interaction)
            if efficiency_routine and len(neighbors[node]) > max_interaction_size:
                # If not all interactions were considered, add the complete the neighborhood interaction
                # This is required for efficiency later on
                incomplete_neighborhoods.add(neighbors[node])

        # Convert neighborhoods into coalition matrix
        incomplete_neighborhoods_matrix, incomplete_neighborhoods_lookup = (
            self._convert_to_coalition_matrix(
                incomplete_neighborhoods, lookup_shift=len(moebius_interactions)
            )
        )
        # Convert collected coalitions into coalition matrix
        moebius_coalition_matrix, moebius_coalition_lookup = self._convert_to_coalition_matrix(
            moebius_interactions
        )
        # Evaluate the (stacked) coalition matrix
        all_coalitions = np.vstack((moebius_coalition_matrix, incomplete_neighborhoods_matrix))
        masked_predictions = self.game(all_coalitions)

        self.last_n_model_calls = np.shape(all_coalitions)[0]

        moebius_coefficients = self.compute_moebius_transform(
            coalitions=moebius_interactions,
            coalition_predictions=masked_predictions,
            coalition_lookup=moebius_coalition_lookup,
        )

        moebius_coefficients.sparsify()

        if efficiency_routine:
            # Add neighborhood interactions
            incomplete_neighborhoods_sorted = sorted(incomplete_neighborhoods, key=len)
            incomplete_neighborhoods_size = len(incomplete_neighborhoods_lookup)
            additional_moebius_coefficients_values = np.zeros(incomplete_neighborhoods_size)
            additional_moebius_coefficients_lookup = {}

            for i, neighborhood_coalition in enumerate(incomplete_neighborhoods_sorted):
                # Compute efficiency gap due to incomplete neighborhood
                sum_of_moebius_coefficients = 0
                for interaction in powerset(neighborhood_coalition, max_size=max_interaction_size):
                    sum_of_moebius_coefficients += moebius_coefficients[interaction]
                for other_incomplete_moebius in incomplete_neighborhoods_sorted:
                    if set(other_incomplete_moebius).issubset(set(neighborhood_coalition)) and len(
                        other_incomplete_moebius
                    ) < len(neighborhood_coalition):
                        sum_of_moebius_coefficients += additional_moebius_coefficients_values[
                            additional_moebius_coefficients_lookup[other_incomplete_moebius]
                        ]
                # match with prediction of neighborhood
                gap = (
                    masked_predictions[incomplete_neighborhoods_lookup[neighborhood_coalition]]
                    - sum_of_moebius_coefficients
                )
                # Assign gap to neighborhood moebius coefficient
                additional_moebius_coefficients_values[i] = gap
                additional_moebius_coefficients_lookup[neighborhood_coalition] = i

                if len(additional_moebius_coefficients_lookup) > 0:
                    # Maintain efficiency by adjusting the largest neighborhood set
                    additional_moebius_coefficients_values[-1] = (
                        self._grand_coalition_prediction
                        - np.sum(moebius_coefficients.values)
                        - np.sum(additional_moebius_coefficients_values[:-1])
                    )

            # Store in InteactionValues Object
            additional_moebius_coefficients = InteractionValues(
                values=additional_moebius_coefficients_values,
                index="Moebius",
                max_order=self.n_players,
                min_order=0,
                n_players=self.n_players,
                interaction_lookup=additional_moebius_coefficients_lookup,
                baseline_value=0,
            )

            final_moebius_coefficients = moebius_coefficients + additional_moebius_coefficients
        else:
            final_moebius_coefficients = moebius_coefficients

        converter = MoebiusConverter(
            N=self._grand_coalition_set,
            moebius_coefficients=final_moebius_coefficients,
        )

        interactions = converter.moebius_to_shapley_interaction(order=order, index="k-SII")
        interactions.sparsify()

        return final_moebius_coefficients, interactions

    def _sii_weight(self, n, q, t, s):
        rslt = 0
        for i in range(n - q + 1):
            sii_weight = 1 / ((n - s + 1) * binom(n - s, t + i))
            rslt += binom(n - q, i) * sii_weight
        return rslt

    def _compute_sii(self, neighbors, order, masked_predictions, coalition_lookup):
        n_interactions = 0
        for node in powerset(self.N, min_size=1, max_size=order):
            n_interactions += 1

        rslt = np.zeros(n_interactions)

        for T in powerset(neighbors):
            t = len(T)
            T_val = masked_predictions[coalition_lookup[T]]
            for S in powerset(neighbors, min_size=1, max_size=order):
                s_cap_t = len(set(T).intersection(set(S)))
                rslt[S] += (
                    (-1) ** (len(S) - s_cap_t)
                    * self._sii_weight(self.n, len(neighbors), t - s_cap_t, len(S))
                    * T_val
                )
        return rslt

    def _get_discrete_derivative(self, S, T, masked_predictions, coalition_lookup):
        """
        Computes the Discrete Derivative of S given T
        S: Subset of N as set or tuple
        T: Subset of N as set or tuple
        """
        rslt = 0
        s = len(S)
        for L in powerset(S):
            l = len(L)
            pos = coalition_lookup[tuple(sorted(set(T).union(set(L))))]
            rslt += (-1) ** (s - l) * masked_predictions[pos]
        return rslt
