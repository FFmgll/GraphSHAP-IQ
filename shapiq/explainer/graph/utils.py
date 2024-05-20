"""This utility module contains functions to get the instances to explain for different datasets."""

import os
from typing import Union

import torch
from torch_geometric.loader import DataLoader

from shapiq.explainer.graph.graph_datasets import CustomTUDataset
from shapiq.explainer.graph.graph_models import GNN

GRAPH_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_datasets")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "graph_prediction")
from .graph_datasets import WaterQuality


def _compute_baseline_value(x_graph):
	# This function computes the baseline value for the masked features, i.e. mean over nodes.
	return x_graph.x.mean(0)


def get_water_quality_graph():
	ds = WaterQuality(subset="train")
	ds = DataLoader(ds, batch_size=1, shuffle=True, pin_memory_device="cpu")
	graph = next(iter(ds))
	return [graph]


def get_tu_instances(name):
	"""Get the instances to explain for the given TU dataset."""
	dataset = CustomTUDataset(
			root=GRAPH_DATASETS_DIR,
			name=name,
			seed=1234,
			split_sizes=(0.8, 0.1, 0.1),
			)
	loader = DataLoader(dataset, shuffle=False)
	try:
		all_samples_to_explain = []
		for data in loader:
			for i in range(data.num_graphs):
				all_samples_to_explain.append(data[i])
		return all_samples_to_explain
	except TypeError:
		return dataset.graphs  # or return dataset.explanations


def load_graph_model_architecture(
		model_type: str,
		dataset_name: str,
		n_layers: int,
		hidden: Union[int, bool] = True,
		node_bias: bool = True,
		graph_bias: bool = True,
		dropout: bool = True,
		batch_norm: bool = True,
		jumping_knowledge: bool = True,
		device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
		) -> tuple[torch.nn.Module, str]:
	"""Loads a graph model architecture, whose weights have to be further trained or loaded.

    Args:

    Returns:
        The loaded model."""
	if dataset_name in [
			"AIDS",
			"DHFR",
			"COX2",
			"BZR",
			"MUTAG",
			"BENZENE",
			"PROTEINS",
			"ENZYMES",
			"Mutagenicity",
			"FluorideCarbonyl",
			"Benzene",
			"AlkaneCarbonyl",
			]:
		dataset = CustomTUDataset(
				root=GRAPH_DATASETS_DIR, name=dataset_name, seed=1234, split_sizes=(0.8, 0.1, 0.1)
				)
		try:
			num_nodes_features = dataset.graphs.num_node_features
			num_classes = dataset.graphs.num_classes
		except AttributeError:
			num_nodes_features = dataset.graphs[0].num_node_features
			num_classes = 2
	else:
		raise Exception("Dataset not found. It has to be downloaded first.")

	if hidden is True:
		# Load the best hyperparameters (for now only hidden size)
		hidden = _best_hyperparameters[model_type][dataset_name]["n_layers"][str(n_layers)][
			"hidden"
		]
	else:
		hidden = 16

	if model_type in ["GCN", "GIN", "GAT"]:
		model = GNN(
				model_type=model_type,
				in_channels=num_nodes_features,
				hidden_channels=hidden,
				out_channels=num_classes,
				n_layers=n_layers,
				node_bias=node_bias,
				graph_bias=graph_bias,
				dropout=dropout,
				batch_norm=batch_norm,
				jumping_knowledge=jumping_knowledge,
				).to(device)
		model.node_model.to(device)
	else:
		raise ValueError("Model type not supported.")

	model_id = "_".join(
			[
					model_type,
					dataset_name,
					str(n_layers),
					str(node_bias),
					str(graph_bias),
					str(hidden),
					str(dropout),
					str(batch_norm),
					str(jumping_knowledge),
					]
			)

	return model, model_id


def load_graph_model(
		model_type: str,
		dataset_name: str,
		n_layers: int,
		hidden: Union[int, bool] = True,
		node_bias: bool = True,
		graph_bias: bool = True,
		dropout: bool = True,
		batch_norm: bool = True,
		jumping_knowledge: bool = True,
		device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
		) -> torch.nn.Module:
	"""Loads a pre-trained graph model from disk with the given configuration.

    Args:
        model_type: The type of the model to load. (e.g. "GCN", "GIN")
        dataset_name: The name of the dataset to load the model for. (e.g. "Mutagenicity")
        n_layers: The number of layers of the model. Can be 1, 2, 3, or 4.
        hidden: The hidden size of the model. If True, the best hyperparameter is loaded.
        node_bias: Whether to use node bias. Default is True.
        graph_bias: Whether to use graph bias. Default is True.
        dropout: Whether to use dropout. Default is True.
        batch_norm: Whether to use batch normalization. Default is True.
        jumping_knowledge: Whether to use jumping knowledge. Default is True.
        device: The device to load the model on.
    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the model file is not found.
    """
	try:
		model, model_id = load_graph_model_architecture(
				model_type=model_type,
				dataset_name=dataset_name,
				n_layers=n_layers,
				hidden=hidden,
				node_bias=node_bias,
				graph_bias=graph_bias,
				dropout=dropout,
				batch_norm=batch_norm,
				jumping_knowledge=jumping_knowledge,
				device=device,
				)

		# Construct the path to the target directory
		target_dir = os.path.join(MODEL_DIR, model_type, dataset_name)
		save_path = os.path.join(target_dir, model_id + ".pth")

		# Load the model (if it exists and it has been trained)
		model.load_state_dict(torch.load(save_path, map_location=device))

	except FileNotFoundError as error:
		raise FileNotFoundError("Model not found. Are you sure you trained the model?") from error
	return model


def get_explanation_instances(dataset_name):
	"""Get the instances to explain for the given dataset."""
	if dataset_name in [
			"AIDS",
			"DHFR",
			"COX2",
			"BZR",
			"MUTAG",
			"BENZENE",
			"PROTEINS",
			"ENZYMES",
			"Mutagenicity",
			"FluorideCarbonyl",
			"Benzene",
			"AlkaneCarbonyl",
			]:
		all_samples_to_explain = get_tu_instances(dataset_name)
		return all_samples_to_explain
	raise ValueError("Dataset not supported.")


# Helper container with all the stored best configurations
_best_hyperparameters = {
		"GCN": {
				"AIDS": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 128},
								"3": {"hidden": 128},
								"4": {"hidden": 128},
								}
						},
				"AlkaneCarbonyl": {
						"n_layers": {
								"1": {"hidden": 64},
								"2": {"hidden": 32},
								"3": {"hidden": 64},
								"4": {"hidden": 128},
								}
						},
				"Benzene": {
						"n_layers": {
								"1": {"hidden": 64},
								"2": {"hidden": 64},
								"3": {"hidden": 128},
								"4": {"hidden": 0},
								}
						},
				"DHFR": {
						"n_layers": {
								"1": {"hidden": 64},
								"2": {"hidden": 128},
								"3": {"hidden": 32},
								"4": {"hidden": 64},
								}
						},
				"COX2": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 128},
								"3": {"hidden": 128},
								"4": {"hidden": 32},
								}
						},
				"BZR": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 64},
								"3": {"hidden": 128},
								"4": {"hidden": 128},
								},
						},
				"FluorideCarbonyl": {
						"n_layers": {
								"1": {"hidden": 16},
								"2": {"hidden": 64},
								"3": {"hidden": 32},
								"4": {"hidden": 0},
								}
						},
				"PROTEINS": {
						"n_layers": {
								"1": {"hidden": 64},
								"2": {"hidden": 64},
								"3": {"hidden": 128},
								"4": {"hidden": 32},
								},
						},
				"ENZYMES": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 64},
								"3": {"hidden": 32},
								"4": {"hidden": 64},
								},
						},
				"MUTAG": {},
				"Mutagenicity": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 64},
								"3": {"hidden": 128},
								"4": {"hidden": 64},
								}
						},
				},
		"GIN": {
				"AlkaneCarbonyl": {
						"n_layers": {
								"1": {"hidden": 16},
								"2": {"hidden": 64},
								"3": {"hidden": 128},
								"4": {"hidden": 0},
								}
						},
				"Benzene": {
						"n_layers": {
								"1": {"hidden": 16},
								"2": {"hidden": 128},
								"3": {"hidden": 128},
								"4": {"hidden": 0},
								},
						},
				"BZR": {
						"n_layers": {
								"1": {"hidden": 64},
								"2": {"hidden": 64},
								"3": {"hidden": 128},
								"4": {"hidden": 32},
								}
						},
				"COX2": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 32},
								"3": {"hidden": 128},
								"4": {"hidden": 128},
								}
						},
				"ENZYMES": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 32},
								"3": {"hidden": 128},
								"4": {"hidden": 128},
								},
						},
				"FluorideCarbonyl": {
						"n_layers": {
								"1": {"hidden": 64},
								"2": {"hidden": 32},
								"3": {"hidden": 32},
								"4": {"hidden": 0},
								}
						},
				"Mutagenicity": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 32},
								"3": {"hidden": 32},
								"4": {"hidden": 64},
								},
						},
				"PROTEINS": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 128},
								"3": {"hidden": 32},
								"4": {"hidden": 32},
								},
						},
				},
		"GAT": {
				"AlkaneCarbonyl": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 16},
								"3": {"hidden": 64},
								"4": {"hidden": 0},
								}
						},
				"Benzene": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 128},
								"3": {"hidden": 64},
								"4": {"hidden": 0},
								},
						},
				"BZR": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 128},
								"3": {"hidden": 128},
								"4": {"hidden": 8},
								},
						},
				"COX2": {
						"n_layers": {
								"1": {"hidden": 64},
								"2": {"hidden": 32},
								"3": {"hidden": 32},
								"4": {"hidden": 32},
								}
						},
				"ENZYMES": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 32},
								"3": {"hidden": 64},
								"4": {"hidden": 32},
								}
						},
				"FluorideCarbonyl": {
						"n_layers": {
								"1": {"hidden": 16},
								"2": {"hidden": 16},
								"3": {"hidden": 16},
								"4": {"hidden": 0},
								}
						},
				"Mutagenicity": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 64},
								"3": {"hidden": 128},
								"4": {"hidden": 64},
								}
						},
				"PROTEINS": {
						"n_layers": {
								"1": {"hidden": 128},
								"2": {"hidden": 32},
								"3": {"hidden": 128},
								"4": {"hidden": 32},
								},
						}
				},
		}
