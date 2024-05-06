import warnings
from typing import Callable, Union
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from shapiq.games.base import Game
from typing import Optional


class GraphGame(Game):
    """The GraphGame game class.

    The `GraphGame` game is a game that performs local explanation on a torch Graph Neural Network based on the graph nodes.
    The game evaluates deploys a graph-specific removal technique for the model's prediction on node subsets.
    The GraphGame may be used for node prediction or graph prediction, which should be included in the model

    Args:
        x: The graph to explain. Should be a torch tensor representing a graph.
        model: The torch graph neural network to explain as a callable function expecting data points as input and
            returning the model's predictions. The input should be a torch tensor representing a graph or batch of graphs.
        masking_mode: Masking technique implemented for node-removal.
        normalize: If True, then each prediction is normalized by the baseline prediction, where all nodes are masked.
        class_id: The position of the response to explain, e.g. the class

    Attributes:
        x_graph: The graph to explain. Should be a torch tensor representing a graph.
        y_index: The position of the response to explain, e.g. the class
        model: The torch graph neural network to explain.
        normalize_predictions: If True, then each prediction is normalized by the baseline prediction, where all nodes are masked.
        masking_mode: Masking technique implemented for node-removal.
        n: The number of nodes.
        N: The set of nodes.
        edge_index: The graph structure in sparse edge_index representation.
        baseline_value: The baseline value representing the prediction with an empty graph.
    """

    def __init__(
        self,
        model: Callable,
        x_graph: Data,
        max_neighborhood_size: int,
        class_id: np.ndarray,
        masking_mode: str = "feature-removal",
        normalize: bool = True,
        baseline: Optional[np.ndarray] = None,
    ) -> None:
        if baseline is None:
            warnings.warn("Baseline is not provided, baseline will be initialized as zero...")
        self.model = model
        self.max_neighborhood_size = max_neighborhood_size
        self.x_graph = x_graph.clone()
        self.baseline = baseline
        self.output_dim = 1
        self.edge_index = self.x_graph.edge_index.detach().numpy()
        # Set masking function
        self.masking_mode = masking_mode
        if self.masking_mode == "feature-removal":
            self.masking = self.mask_input
        if self.masking_mode == "node-removal":
            self.masking = self.mask_nodes
        self.y_index = class_id
        # Compute emptyset prediction
        normalization_value = self.value_function(np.zeros(len(x_graph.x)))
        # call the super constructor
        super().__init__(
            n_players=len(x_graph.x), normalize=normalize, normalization_value=normalization_value
        )
        self._grand_coalition_set = set(range(self.n_players))

    def _precompute_baseline_value(self, x_graph: Data, y_index: np.ndarray) -> float:
        # Mask all nodes for emptyset prediction
        x_graph_empty = self.masking(np.zeros(len(x_graph.x)))
        baseline_value = self.model(x_graph_empty.x, x_graph_empty.edge_index, x_graph_empty.batch)
        return baseline_value.detach().numpy()[:, y_index]

    def mask_input(self, coalition: np.ndarray) -> Data:
        """The masking procedure for feature-removal. Masks all feature values of masked nodes.

        Args:
            coalition: A binary numpy array containing the masking.

        Returns: The masked x_graph for the coalition as a graph tensor.
        """
        x_masked = self.x_graph.clone()
        if self.baseline is None:
            x_masked.x *= torch.tensor(coalition.reshape((-1, 1)), dtype=torch.float32)
        else:
            x_masked.x = x_masked.x*torch.tensor(coalition.reshape((-1, 1)), dtype=torch.float32)+ self.baseline*torch.tensor((1-coalition).reshape((-1, 1)), dtype=torch.float32)
        return x_masked

    def mask_nodes(self, coalition: np.ndarray) -> Data:
        """The masking procedure for node-removal. Removes all masked nodes from x_graph.

        Args:
            coalitions: A binary numpy array containing the maskings.

        Returns: The masked x_graph for each coalition as graph tensors.
        """
        if np.sum(coalition) == self.n_players:
            # Special case when all nodes should be removed. Results in one node with zero features.
            data = self.x_graph.clone()
            masked_graph = data.subgraph(
                coalition
            )  # We remove the specific nodes and all edges connected to them
            # set node feature to zero
            masked_graph.x.zero_()
            masked_graph.validate()
        else:
            data = self.x_graph.clone()
            masked_graph = data.subgraph(
                coalition
            )  # We remove the specific nodes and all edges connected to them
            masked_graph.validate()  # This is important, raises an error if the Data class is corrupted by masking
        return masked_graph

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """The value function used to call the model with an array of coalitions.

        Args:
            coalitions: A binary numpy array containing the maskings.

        Returns: The masked predictions for each coalition as numpy array.
        """
        # Initialize masked graph list
        masked_graph_list = []
        if len(np.shape(coalitions)) == 1:
            coalitions = coalitions.reshape(1, -1)
        n_masked_graphs = np.shape(coalitions)[0]

        # Created masked graph, by masking nodes
        for i in range(n_masked_graphs):
            # Create masked graph
            x_masked = self.masking(coalitions[i])
            masked_graph_list.append(x_masked)

        # Create a data batch of graphs
        graph_list = [Data(**graph) for graph in masked_graph_list]
        masked_batch = Batch.from_data_list(graph_list)

        # Call model once using the batch
        masked_predictions = self.model(
            x=masked_batch.x,
            edge_index=masked_batch.edge_index,
            batch=masked_batch.batch,
        )
        return masked_predictions.detach().numpy()[:, self.y_index]


class GraphNodeGame(Game):
    """The GraphNodeGame game class.

    Args:
        x: The graph to explain. Should be a torch tensor representing a graph.
        model: The torch graph neural network to explain as a callable function expecting data points as input and
            returning the model's predictions. The input should be a torch tensor representing a graph or batch of graphs.
        masking_mode: Masking technique implemented for node-removal.
        normalize: If True, then each prediction is normalized by the baseline prediction, where all nodes are masked.
        y_index: The position of the response to explain, e.g. the class

    Attributes:
        x_graph: The graph to explain. Should be a torch tensor representing a graph.
        y_index: The position of the response to explain, e.g. the class
        model: The torch graph neural network to explain.
        normalize_predictions: If True, then each prediction is normalized by the baseline prediction, where all nodes are masked.
        masking_mode: Masking technique implemented for node-removal.
        n: The number of nodes.
        N: The set of nodes.
        edge_index: The graph structure in sparse edge_index representation.
        baseline_value: The baseline value representing the prediction with an empty graph.
    """

    def __init__(
        self,
        model: Callable,
        max_neighborhood_size: int,
        x_graph: Data,
        class_labels: np.ndarray,
        node_id: Optional[int] = None,
        masking_mode: str = "feature-removal",
        normalize: bool = True,
    ) -> None:
        self.model = model
        self.max_neighborhood_size = max_neighborhood_size
        # The complete graph used to call the model
        self.x_graph = x_graph.clone()
        self.num_nodes = self.x_graph.num_nodes
        self.edge_index = self.x_graph.edge_index.detach().numpy()
        # The label of the node
        self.class_labels = class_labels
        self.node_id = node_id
        if node_id is None:
            self.output_dim = self.num_nodes
        else:
            self.output_dim = 1

        # Set masking function
        self.masking_mode = masking_mode
        if self.masking_mode == "feature-removal":
            self.masking = self.mask_input

        # Compute emptyset prediction
        baseline_value = self.value_function(np.zeros(self.num_nodes))
        # call the super constructor
        super().__init__(
            n_players=self.num_nodes, normalize=normalize, normalization_value=baseline_value
        )
        self._grand_coalition_set = set(range(self.n_players))

    def mask_input(self, coalition: np.ndarray) -> Data:
        """The masking procedure for feature-removal. Masks all feature values of masked nodes.

        Args:
            coalition: A binary numpy array containing the masking.

        Returns: The masked x_graph for the coalition as a graph tensor.
        """
        x_masked = self.x_graph.clone()
        x_masked.x *= torch.tensor(coalition.reshape((-1, 1)), dtype=torch.float32)
        return x_masked

    def mask_nodes(self, coalition: np.ndarray) -> Data:
        """The masking procedure for node-removal. Removes all masked nodes from x_graph.

        Args:
            coalitions: A binary numpy array containing the maskings.

        Returns: The masked x_graph for each coalition as graph tensors.
        """
        if np.sum(coalition) == self.n_players:
            # Special case when all nodes should be removed. Results in one node with zero features.
            data = self.x_graph.clone()
            masked_graph = data.subgraph(
                coalition
            )  # We remove the specific nodes and all edges connected to them
            # set node feature to zero
            masked_graph.x.zero_()
            masked_graph.validate()
        else:
            data = self.x_graph.clone()
            masked_graph = data.subgraph(
                coalition
            )  # We remove the specific nodes and all edges connected to them
            masked_graph.validate()  # This is important, raises an error if the Data class is corrupted by masking
        return masked_graph

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """The value function used to call the model with an array of coalitions.

        Args:
            coalitions: A binary numpy array containing the maskings.

        Returns: The masked predictions for each coalition as numpy array.
        """
        if len(np.shape(coalitions)) == 1:
            coalitions = coalitions.reshape(1, -1)
        n_coalitions = np.shape(coalitions)[0]

        coalitions_worth = np.zeros((n_coalitions, self.num_nodes))
        for pos in zip(range(n_coalitions)):
            x_masked = self.masking(coalitions[pos])
            masked_graph_prediction = self.model(x_masked)[
                np.arange(self.num_nodes), self.class_labels
            ]
            coalitions_worth[pos, :] = masked_graph_prediction.detach().numpy()

        if self.node_id is None:
            # Return all nodes
            return coalitions_worth
        else:
            # Return single node as vector
            return coalitions_worth[:, self.node_id].reshape(-1)
