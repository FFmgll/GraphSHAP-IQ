"""
Here we train a GNN for Node Classification. See below name guard for the main function and usage.
Datasets in order of complexity:
- Cora
TODO:
- Citeseer
- Pubmed
(other possible datasets: Chameleon, Squirrel, Actor, Coauthor, CS, Physics)
Models:
- GCN
TODO:
- GAT
- SAGE
- GIN

Besides, later I will set up a hyperparameter search for the best model and dataset combination :-)
"""
from pathlib import Path
import os
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import torch_geometric.nn.models as pyg_models
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


# Load dataset
def dataset_node_classification(dataset_name: str = "Cora"):
	"This loads the dataset for node classification with the standard splits"
	dataset_path = Path(__file__).resolve().parents[3] / "data"  # Path to the data folder in main directory
	dataset = Planetoid(root=dataset_path, name=dataset_name)
	return dataset


class NodeClassificationGNN(L.LightningModule):
	"""Node Classification GNN.
	Load from checkpoint if exists, otherwise trains it.
	Args:
		- dataset: PyG dataset
		- model_name: str, name of the model (GCN, GAT, SAGE, GIN)
		- num_layers: int, number of hidden layers
	returns:
		- model: PyG model
		- train_loader: PyG DataLoader
		- val_loader: PyG DataLoader
		- test_loader: PyG DataLoader
	"""

	def __init__(self, model_name: str = "GCN", **model_kwargs):
		super().__init__()
		self.save_hyperparameters()
		self.model = getattr(pyg_models, model_name)(**model_kwargs)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1, weight_decay=2e-3)
		self.loss = torch.nn.CrossEntropyLoss()

	def forward(self, data):
		"""Forward pass of the model. Returns the logits."""
		out = self.model(data.x, data.edge_index)
		return out

	def training_step(self, batch, batch_idx):
		mask = batch.train_mask  # Get the mask for the training nodes
		out = self.forward(batch)
		loss = self.loss(out[mask], batch.y[mask])
		accuracy = (out[mask].argmax(dim=-1) == batch.y[mask]).sum().float() / mask.sum()
		self.log("train_loss", loss)
		self.log("train_accuracy", accuracy)
		return loss

	def validation_step(self, batch, batch_idx):
		mask = batch.val_mask
		out = self.forward(batch)
		accuracy = (out[mask].argmax(dim=-1) == batch.y[mask]).sum().float() / mask.sum()
		self.log("val_accuracy", accuracy)

	def test_step(self, batch, batch_idx):
		mask = batch.test_mask
		out = self.forward(batch)
		accuracy = (out[mask].argmax(dim=-1) == batch.y[mask]).sum().float() / mask.sum()
		self.log("test_accuracy", accuracy)

	def configure_optimizers(self):
		return self.optimizer


def get_node_classifier(dataset_name: str = "Cora", model_name: str = "GCN", num_layers: int = 3,
						hidden_channels: int = 16, **model_kwargs):
	"""Get the node classifier model, train it if it doesn't exist and return it with the data"""
	dataset = dataset_node_classification(dataset_name)
	data_loader = DataLoader(dataset, batch_size=1)  # There is only 1 graph

	model_path = Path(__file__).resolve().parent / "ckpt" / f"{model_name}_{dataset_name}_{num_layers}.ckpt"

	trainer = L.Trainer(max_epochs=200,
						default_root_dir=model_path,
						#callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor="val_accuracy")],
						enable_progress_bar=False,
						log_every_n_steps=1)

	#if model_path.exists():
	#	print("Model found. Loading...")
	#	model = NodeClassificationGNN.load_from_checkpoint(model_path)
	#else:
	L.seed_everything(42)
	model = NodeClassificationGNN(model_name=model_name, in_channels=dataset.num_node_features,
								  out_channels=dataset.num_classes,
								  num_layers=num_layers, hidden_channels=16, **model_kwargs)
	print(f"Dataset \'{dataset_name}\': Model {model_name} with {num_layers=} not found. Training...")

	trainer.fit(model, data_loader, data_loader)
	model = NodeClassificationGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

	# Test best model on the test set
	batch = next(iter(data_loader))
	batch = batch.to(model.device)
	out = model(batch)
	# Validation accuracy
	val_acc = (out[batch.val_mask].argmax(dim=-1) == batch.y[batch.val_mask]).sum().float() / batch.val_mask.sum()
	# Training accuracy
	train_acc = (out[batch.train_mask].argmax(dim=-1) == batch.y[
		batch.train_mask]).sum().float() / batch.train_mask.sum()
	# Test accuracy
	test_acc = (out[batch.test_mask].argmax(dim=-1) == batch.y[batch.test_mask]).sum().float() / batch.test_mask.sum()
	print(f"Train Acc: {100 * train_acc:.3f}%, Val Acc: {100 * val_acc:.3f}%, , Test Acc: {100 * test_acc:.3f}%}")

	return model, data_loader


if __name__ == "__main__":
	# Train a GNN for Node Classification
	my_beautiful_gnn, data_loader = get_node_classifier(dataset_name="Cora", model_name="GCN", num_layers=2)
	# Evaluate the model
	my_beautiful_gnn.eval()
	batch = next(iter(data_loader))  # Take the 1 graph of Cora