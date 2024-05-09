"""This method trains the GNN architectures and stores them under /ckpt.
Naming convention is MODELTYPE_DATASET_NLayers_NodeBias_GraphBias_HiddenUnits_Dropout_BatchNorm_JumpingKnowledge,
e.g. GCN_MUTAG_3_False_False_64_True_False_True.
The corresponding directory is MODELTYPE/DATASET, e.g. GCN/MUTAG"""

import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from graphxai_local.datasets import (
	MUTAG,
)
from shapiq.explainer.graph.graph_datasets import CustomTUDataset
from shapiq.explainer.graph.utils import MODEL_DIR, load_graph_model_architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_device(device)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# Path to store the models
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ckpt", "graph_prediction")
DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "graph_datasets")


class EarlyStopper:
	def __init__(self, patience=1, min_delta=0):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.min_validation_loss = float('inf')

	def early_stop(self, validation_loss):
		if validation_loss < self.min_validation_loss:
			self.min_validation_loss = validation_loss
			self.counter = 0
		elif validation_loss > (self.min_validation_loss + self.min_delta):
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False


def get_TU_dataset(device, name):
	"""Get the TU dataset by name"""
	# Load dataset
	if not os.path.exists(DATASET_PATH):
		os.makedirs(DATASET_PATH)
	dataset = CustomTUDataset(
		root=DATASET_PATH, name=name, seed=1234, split_sizes=(0.8, 0.1, 0.1))

	num_nodes_features = dataset.graphs.num_node_features
	num_classes = dataset.graphs.num_classes
	# TODO: Consider using stratified sampling for multi-class classification

	train_loader = DataLoader(dataset[dataset.train_index], batch_size=32, shuffle=True)
	val_loader = DataLoader(dataset[dataset.val_index], batch_size=32, shuffle=False)
	test_loader = DataLoader(dataset[dataset.test_index], batch_size=32, shuffle=False)

	return train_loader, val_loader, test_loader, num_nodes_features, num_classes


def train_and_store(model, train_loader, val_loader, test_loader, save_path):
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss() if model.out_channels > 1 else torch.nn.BCELoss()
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,
														   threshold=1e-3)
	early_stopper = EarlyStopper(patience=50)

	model_name = model.__class__.__name__
	dataset_name = train_loader.dataset.name
	log_dir = Path("shapiq", "explainer", "graph", "ckpt", "training_logs",
				   "graph_prediction", model_name, dataset_name).resolve()
	log_dir.mkdir(parents=True, exist_ok=True)
	writer = SummaryWriter(log_dir=log_dir)

	# Train and test functions
	def train(graph_model):
		graph_model.train()
		for data in train_loader:
			data = data.to(device)
			# Iterate in batches over the training dataset.
			out = graph_model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
			loss = criterion(out, data.y)  # Compute the loss.
			loss.backward()  # Derive gradients.
			optimizer.step()  # Update parameters based on gradients.
			optimizer.zero_grad()  # Clear gradients.

	@torch.no_grad()
	def test(loader, graph_model):
		graph_model.eval()
		correct = 0
		loss = 0
		for data in loader:  # Iterate in batches over the training/test dataset.
			data = data.to(device)
			out = graph_model(data.x, data.edge_index, data.batch)
			loss += criterion(out, data.y).item()
			pred = out.argmax(dim=1)  # Use the class with highest probability.
			correct += int((pred == data.y).sum())  # Check against ground-truth labels.
		return correct / len(loader.dataset), loss  # Derive ratio of correct predictions.

	# Train model
	best_test_acc = 0
	best_val_acc = 0
	for epoch in range(1, 500):
		train(graph_model=model)  # uncomment to train
		train_acc, train_loss = test(train_loader, graph_model=model)
		val_acc, val_loss = test(val_loader, graph_model=model)
		scheduler.step(val_loss)
		test_acc, test_loss = test(test_loader, graph_model=model)
		print(f"Epoch: {epoch}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f},"
			  f" Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

		# Save best model on validation set
		if epoch == 1 or val_acc >= best_val_acc:
			best_val_acc = val_acc
			best_test_acc = test_acc
			torch.save(model.state_dict(), save_path)
			print(f"Best model saved at epoch {epoch}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

		if early_stopper.early_stop(val_loss):
			print(f"Early stopping at epoch {epoch}")
			break

	writer.add_hparams(
			{'model': model_name, 'dataset': dataset_name, 'n_layers': model.n_layers, 'hidden': model.hidden_channels,
			 'node_bias': model.node_bias, 'graph_bias': model.graph_bias, 'dropout': model.dropout,
			 'batch_norm': model.batch_norm,
			 'jumping_knowledge': model.jumping_knowledge},
			{'hparam/val_acc': best_val_acc, 'hparam/train_acc': train_acc, 'hparam/test_acc': best_test_acc})
	writer.close()


def train_gnn(dataset_name, model_type, n_layers, node_bias=True, graph_bias=True, hidden=True, dropout=True, batch_norm=True, jumping_knowledge=True,
			  enforce_retrain=False):
	if dataset_name in ["AIDS", "DHFR", "COX2", "BZR", "MUTAG", "BENZENE", "PROTEINS", "ENZYMES", "Mutagenicity"]:
		train_loader, val_loader, test_loader, num_nodes_features, num_classes = get_TU_dataset(device,
																								dataset_name)
	else:
		raise Exception("Dataset not found")

	model, model_id = load_graph_model_architecture(model_type, dataset_name, n_layers, hidden, node_bias, graph_bias, dropout,
										  batch_norm, jumping_knowledge, device)
	# Construct the path to the target directory
	target_dir = Path(MODEL_DIR, model_type, dataset_name).resolve()
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

if __name__ == "__main__":
	model, model_id = train_gnn("Mutagenicity", "GCN", 1)