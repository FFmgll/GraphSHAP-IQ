"""This module contains the GraphSHAP-IQ class"""

from typing import Union, Optional
import multiprocessing as mp
from shapiq.games.benchmark.local_xai.benchmark_graph import GraphGame, GraphNodeGame

import numpy as np
import copy
from shapiq.utils import powerset
from shapiq.interaction_values import InteractionValues
from scipy.special import binom


class L_Shapley:
    def __init__(self, game: Union[GraphGame, GraphNodeGame], max_budget: int):
        self._last_n_model_calls: Optional[int] = None
        self.edge_index = game.edge_index
        self.n_players = game.n_players
        self.sparsify_threshold = 10e-10
        self._grand_coalition_set = game._grand_coalition_set
        self.n_jobs = mp.cpu_count() - 1
        self.output_dim = game.output_dim
        self.game = game
        self._grand_coalition_prediction = game(np.ones(self.game.n_players))
        self.max_budget = max_budget

    def _get_neighborhoods(self):
        """Computes the neighborhoods of each node and caps the max_interaction_size at the size of
        the largest neighborhood.

        Returns:
            neighbors: A dictionary containing all neighbors of each node
            max_interaction_size: max_interaction_size capped at the largest neighborhood size
        """
        neighbors = {}
        max_size_neighbors = 0
        for neighbor_id in self._grand_coalition_set:
            neighbor_list = self._get_k_neighborhood(neighbor_id)
            max_size_neighbors = max(max_size_neighbors, len(neighbor_list))
            neighbors[neighbor_id] = neighbor_list
        return neighbors, max_size_neighbors

    def _get_k_neighborhood(self, node):
        neighbors = set()
        queue = [(node, 0)]
        visited = set([node])
        while queue:
            curr_node, dist = queue.pop(0)
            if dist > self.max_neighborhood_size:
                break
            if dist <= self.max_neighborhood_size:
                neighbors.add(curr_node)
            if dist < self.max_neighborhood_size:
                # Find neighbors of current node
                for edge in self.edge_index.T:
                    if edge[0] == curr_node and edge[1] not in visited:
                        queue.append((edge[1], dist + 1))
                        visited.add(edge[1])
        return tuple(sorted(neighbors))

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
        self, max_interaction_size: int, break_on_exceeding_budget: bool
    ) -> tuple[InteractionValues, bool]:

        self.max_neighborhood_size = max_interaction_size
        self.neighbors, self.max_size_neighbors = self._get_neighborhoods()
        # Cap max_interaction_size
        max_interaction_size = min(self.max_size_neighbors, max_interaction_size)
        # Get collection of Möbius interactions to be computed, and complete neighborhoods that are not considered (if
        # efficiency_routine is True)
        coalitions = self._get_all_coalitions(max_interaction_size)
        exceeded_budget = False
        if len(coalitions) > self.max_budget:
            exceeded_budget = True
            if break_on_exceeding_budget:
                raise ValueError("Exceeded budget.")

        # Convert collected coalitions into coalition matrix
        coalition_matrix, coalition_lookup = self._convert_to_coalition_matrix(coalitions)

        # Evaluate the coalition matrix on the GNN
        masked_predictions = self.game(coalition_matrix)
        # Store the model calls
        self.last_n_model_calls = np.shape(coalition_matrix)[0]

        shapley_values = np.zeros(self.n_players)
        shapley_values_lookup = {}
        for player_i in self._grand_coalition_set:
            neighborhood_of_i = self.neighbors[player_i]
            player_i_tuple = tuple([player_i])
            shapley_values_lookup[player_i_tuple] = player_i
            shapley_values[player_i] = self._LShapley_routine(
                neighborhood_of_i,
                player_i_tuple,
                masked_predictions,
                coalition_lookup,
                max_interaction_size,
            )

        int_values = InteractionValues(
            values=shapley_values,
            interaction_lookup=shapley_values_lookup,
            min_order=0,
            max_order=1,
            n_players=self.n_players,
            index="SV",
            baseline_value=float(masked_predictions[coalition_lookup[tuple()]]),
            estimation_budget=self.last_n_model_calls,
        )
        return int_values, exceeded_budget

    def _LShapley_routine(
        self,
        neighborhood_of_i,
        player_i,
        masked_predictions,
        coalition_lookup,
        max_interaction_size,
    ):
        shapley_value = 0
        size_neighborhood = len(neighborhood_of_i)
        for subset in powerset(neighborhood_of_i, max_size=max_interaction_size):
            if set(player_i).issubset(set(subset)):
                no_player_i = tuple(sorted(set(subset) - set(player_i)))
                marginal_contribution = (
                    masked_predictions[coalition_lookup[subset]]
                    - masked_predictions[coalition_lookup[no_player_i]]
                )
                shapley_value += (
                    binom(size_neighborhood - 1, len(subset) - 1) ** (-1) * marginal_contribution
                )

        shapley_value /= size_neighborhood

        return shapley_value

    def _get_all_coalitions(self, max_interaction_size: int) -> (dict, dict):
        """Collects all coalitions that will be evaluated, i.e. coalitions with non-zero Möbius
        transform of maximum size max_interaction_size. If efficiency_routine is True, then all
        neighborhoods are added to the collection.

        Args:
            max_interaction_size: The maximum interaction size considered

        Returns:
            moebius_interactions: The set of Möbius interactions considered
        """
        # Get non-zero Möbius values based on the neighborhood
        incomplete_neighborhoods = set()
        moebius_interactions = set()

        for node in self.neighbors:
            # Collect all non-zero Möbius interactions up to order max_interaction_size
            # For these, game evaluations are required
            for interaction in powerset(self.neighbors[node], max_size=max_interaction_size):
                moebius_interactions.add(interaction)

        return moebius_interactions
