"""This method trains the GNN architectures and stores them under /ckpt.
Naming convention is MODELTYPE_DATASET_NLayers_NodeBias_GraphBias_Dropout_BatchNorm_JumpingKnowledge,
e.g. GCN_MUTAG_3_False_False_True_False_True.
The corresponding directory is MODELTYPE/DATASET, e.g. GCN/MUTAG"""

import os
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from graphxai_local.datasets import (
	MUTAG,
	)  # renamed to avoid conflict with potential installs
from shapiq.explainer.graph.graph_datasets import CustomTUDataset
from shapiq.explainer.graph.graph_models import GCN, GIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def get_MUTAG_dataset(device):
	# Load dataset
	# TODO: Check this dataset and its explanations. We should get a better understanding of what is happening in GraphXAI (Paolo: I remermber they do some pre-processing on Mutagenicity and they call it Mutag)
	dataset = MUTAG(root="", seed=1234, split_sizes=(0.8, 0.1, 0.1))
	dataset.graphs.data.to(device)
	num_nodes_features = dataset.graphs.num_node_features
	num_classes = dataset.graphs.num_classes

	train_loader = DataLoader(dataset[dataset.train_index], batch_size=4, shuffle=True)
	val_loader = DataLoader(
			dataset[dataset.val_index], batch_size=len(dataset.val_index), shuffle=False
			)
	test_loader = DataLoader(
			dataset[dataset.test_index], batch_size=len(dataset.test_index), shuffle=False
			)
	return train_loader, val_loader, test_loader, num_nodes_features, num_classes


def get_TU_dataset(device, name):
	"""Get the TU dataset by name"""
	# Load dataset
	dataset_path = Path("shapiq/explainer/graph/graph_datasets").resolve()
	dataset_path.mkdir(parents=True, exist_ok=True)
	dataset = CustomTUDataset(root=dataset_path, name=name, seed=1234,
							  split_sizes=(0.8, 0.1, 0.1), device=device)

	num_nodes_features = dataset.graphs.num_node_features
	num_classes = dataset.graphs.num_classes

	train_loader = DataLoader(dataset[dataset.train_index], batch_size=32, shuffle=True, generator=torch.Generator(device))
	val_loader = DataLoader(dataset[dataset.val_index], batch_size=32, shuffle=False)
	test_loader = DataLoader(dataset[dataset.test_index], batch_size=32, shuffle=False)

	return train_loader, val_loader, test_loader, num_nodes_features, num_classes


def train_and_store(model, train_loader, val_loader, test_loader, save_path):
	optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss() if model.out_channels > 1 else torch.nn.BCELoss()

	# Train and test functions
	def train(graph_model):
		graph_model.train()
		for data in train_loader:
			data = data.to(device)
			# Iterate in batches over the training dataset.
			out = graph_model(
					data.x, data.edge_index, data.batch
					)  # Perform a single forward pass.
			loss = criterion(out, data.y)  # Compute the loss.
			loss.backward()  # Derive gradients.
			optimizer.step()  # Update parameters based on gradients.
			optimizer.zero_grad()  # Clear gradients.

	def test(loader, graph_model):
		graph_model.eval()
		correct = 0
		for data in loader:  # Iterate in batches over the training/test dataset.
			data = data.to(device)
			out = graph_model(data.x, data.edge_index, data.batch)
			pred = out.argmax(dim=1)  # Use the class with highest probability.
			correct += int((pred == data.y).sum())  # Check against ground-truth labels.
		return correct / len(loader.dataset)  # Derive ratio of correct predictions.

	# set to True to train the model or False to load the best model from the checkpoint
	TRAIN = True

	if TRAIN:
		best_val_acc = 0
		for epoch in range(1, 200):
			train(graph_model=model)  # uncomment to train
			train_acc = test(train_loader, graph_model=model)
			val_acc = test(val_loader, graph_model=model)
			test_acc = test(test_loader, graph_model=model)
			print(f"Epoch: {epoch}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

			# Save best model on validation set
			if epoch == 1 or val_acc >= best_val_acc:
				best_val_acc = val_acc
				torch.save(model.state_dict(), save_path)
				print(f"Best model saved at epoch {epoch}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
	else:
		print(f"Best model loaded from {save_path}")
		model.load_state_dict(torch.load(save_path))
		val_acc = test(val_loader, graph_model=model)
		test_acc = test(test_loader, graph_model=model)
		print(f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

		# Load best model
		model.load_state_dict(torch.load(save_path))

		# Get the accurate samples on the test set (we explain only true positives for now)
		correct_samples = []
		for data in test_loader:  # Only one loop (#num_batches = 1)
			out = model(data.x, data.edge_index, data.batch)
			pred = out.argmax(dim=1)
			correct_samples = data[pred == data.y]


def train_gnn(dataset_name, model_type, n_layers, node_bias, graph_bias, dropout, batch_norm, jumping_knowledge,
			  enforce_retrain=False):

	# if dataset_name == "MUTAG":
	# train_loader, val_loader, test_loader, num_nodes_features, num_classes = get_MUTAG_dataset(device)
	if dataset_name in ["AIDS", "DHFR", "COX2", "BZR", "MUTAG", "BENZENE", "PROTEINS", "ENZYMES", "Mutagenicity"]:
		train_loader, val_loader, test_loader, num_nodes_features, num_classes = get_TU_dataset(device, dataset_name)
	else:
		raise Exception("Dataset not found")

	if model_type == "GCN":
		model = GCN(in_channels=num_nodes_features,
					hidden_channels=64,
					out_channels=num_classes,
					n_layers=n_layers,
					node_bias=node_bias,
					graph_bias=graph_bias,
					dropout=dropout,
					batch_norm=batch_norm,
					jumping_knowledge=jumping_knowledge).to(device)
		model.node_model.to(device)
	elif model_type == "GIN":
		model = GIN(in_channels=num_nodes_features, hidden_channels=64, out_channels=num_classes, n_layers=n_layers,
					graph_bias=graph_bias, node_bias=node_bias).to(device)
		pass
	else:
		raise Exception("Model not found")

	# Compile model with torch if on linux
	if os.name == 'posix':
		model = torch.compile(model)

	model_id = "_".join([model_type, dataset_name, str(n_layers), str(node_bias), str(graph_bias), str(dropout), str(batch_norm), str(jumping_knowledge)])
	# Construct the path to the target directory
	target_dir = Path("shapiq", "explainer", "graph", "ckpt", "graph_prediction", model_type, dataset_name).resolve()
	save_path = Path(target_dir, model_id + ".pth").resolve()

	# Check if the directory exists, if not, create it
	target_dir.mkdir(parents=True, exist_ok=True)

	if not save_path.exists() or enforce_retrain:
		print("Training model: ", model_id)
		train_and_store(model, train_loader, val_loader, test_loader, save_path)
	else:
		print("Loading model: ", model_id)

	model.load_state_dict(torch.load(save_path))
	return model, model_id