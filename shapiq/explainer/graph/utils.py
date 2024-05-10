"""This utility module contains functions to get the instances to explain for different datasets."""

from typing import Any, Union
import os

import torch
from torch_geometric.data import DataLoader

from shapiq.explainer.graph.graph_datasets import CustomTUDataset
from shapiq.explainer.graph.graph_models import GCN, GIN


GRAPH_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_datasets")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "graph_prediction")


def _compute_baseline_value(x_graph):
    # This function computes the baseline value for the masked features, i.e. mean over nodes.
    return x_graph.x.mean(0)


def get_tu_instances(name):
    """Get the instances to explain for the given TU dataset."""
    dataset = CustomTUDataset(
        root=GRAPH_DATASETS_DIR,
        name=name,
        seed=1234,
        split_sizes=(0.8, 0.1, 0.1),
    )
    loader = DataLoader(dataset, shuffle=False)
    all_samples_to_explain = []
    for data in loader:
        for i in range(data.num_graphs):
            all_samples_to_explain.append(data[i])
    return all_samples_to_explain


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
    ]:
        dataset = CustomTUDataset(
            root=GRAPH_DATASETS_DIR, name=dataset_name, seed=1234, split_sizes=(0.8, 0.1, 0.1)
        )
        num_nodes_features = dataset.graphs.num_node_features
        num_classes = dataset.graphs.num_classes
    else:
        raise Exception("Dataset not found. It has to be downloaded first.")

    if hidden is True:
        # Load the best hyperparameters (for now only hidden size)
        hidden = _best_hyperparameters[model_type][dataset_name]["n_layers"][str(n_layers)][
            "hidden"
        ]

    if model_type == "GCN":
        model = GCN(
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
    elif model_type == "GIN":
        model = GIN(
            in_channels=num_nodes_features,
            hidden_channels=64,
            out_channels=num_classes,
            n_layers=n_layers,
            graph_bias=graph_bias,
            node_bias=node_bias,
        ).to(device)
        pass  # TODO: Implement GIN (or general GNN) model + GAT
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
        "PROTEINS",
        "ENZYMES",
        "MUTAG",
        "Mutagenicity",
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
                "4": {"hidden": 128},
            }
        },
        "BZR": {
            "n_layers": {
                "1": {"hidden": 128},
                "2": {"hidden": 32},
                "3": {"hidden": 64},
                "4": {"hidden": 64},
            },
        },
        "PROTEINS": {},
        "ENZYMES": {},
        "MUTAG": {},
        "Mutagenicity": {
            "n_layers": {
                "1": {"hidden": 128},
                "2": {"hidden": 128},
                "3": {"hidden": 128},
                "4": {"hidden": 64},
            }
        },
    },
    "GIN": {},
}
