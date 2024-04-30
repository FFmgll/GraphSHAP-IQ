from pathlib import Path

import lightning as L
import torch
import torch_geometric.nn.models as pyg_models
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torchmetrics.functional import accuracy


# Load dataset
def dataset_node_classification(dataset_name: str = "Cora"):
	"This loads the dataset for node classification with the standard splits"
	dataset_path = Path(__file__).resolve().parents[4] / "data"  # Path to the data folder in main directory
	dataset = Planetoid(root=dataset_path, name=dataset_name)
	return dataset


class NodeClassificationGNN(L.LightningModule):
	"""Node Classification GNN.
	Load from checkpoint if exists, otherwise trains it.
	Args:
		- dataset: PyG dataset
		- model_name: str, name of the model (GCN, GAT, SAGE, GIN)
		- in_channels: int, number of input channels
		- out_channels: int, number of output channels
		- hidden_channels: int, number of hidden channels
		- num_layers: int, number of hidden layers
		- model_kwargs: dict, additional model arguments from torch_geometric.nn.models
	"""

	def __init__(self, model_name: str = "GCN", **model_kwargs):
		super().__init__()
		self.save_hyperparameters()
		self.model = getattr(pyg_models, model_name)(dropout=0.5, **model_kwargs)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
		self.loss = torch.nn.CrossEntropyLoss()

	def forward(self, data):
		"""Forward pass of the model. Returns the logits."""
		out = self.model(data.x, data.edge_index)
		return out

	def training_step(self, batch, batch_idx):
		mask = batch.train_mask  # Get the mask for the training nodes
		out = self.forward(batch)
		loss = self.loss(out[mask], batch.y[mask])
		acc = accuracy(out[mask], batch.y[mask], task="multiclass", num_classes=self.hparams.out_channels)
		self.log("train_loss", loss, batch_size=batch.size(0))
		self.log("train_accuracy", acc, batch_size=batch.size(0))
		return loss

	def validation_step(self, batch, batch_idx):
		mask = batch.val_mask
		out = self.forward(batch)
		acc = accuracy(out[mask], batch.y[mask], task="multiclass", num_classes=self.hparams.out_channels)
		self.log("val_accuracy", acc, batch_size=batch.size(0), prog_bar=False)

	def test_step(self, batch, batch_idx):
		mask = batch.test_mask
		out = self.forward(batch)
		acc = accuracy(out[mask], batch.y[mask], task="multiclass", num_classes=self.hparams.out_channels)
		self.log("test_accuracy", acc, batch_size=batch.size(0), prog_bar=False)

	def configure_optimizers(self):
		return self.optimizer


def get_node_classifier(dataset_name: str = "Cora", model_name: str = "GCN", num_layers: int = 3,
						hidden_channels: int = 16, **model_kwargs):
	"""Get the node classifier model, train it if it doesn't exist and return it with the data"""
	# Data
	dataset = dataset_node_classification(dataset_name)
	data_loader = DataLoader(dataset, batch_size=1, num_workers=4, persistent_workers=True)  # There is only 1 graph

	# Model
	model_dir = Path(__file__).resolve().parents[1] / "ckpt" / "node_prediction" / f"{model_name}" / f"{dataset_name}"
	model_path = model_dir / f"{model_name}_{dataset_name}_{num_layers}.pth"

	checkpoint_callback = ModelCheckpoint(save_weights_only=True,
										  mode='max',
										  monitor="val_accuracy",
										  dirpath=model_dir,
										  filename=f"{model_name}_{dataset_name}_{num_layers}")
	checkpoint_callback.FILE_EXTENSION = ".pth"
	tb_logger = TensorBoardLogger(save_dir=Path(__file__).resolve().parents[1] / "ckpt" / "training_logs")
	trainer = L.Trainer(max_epochs=200,
						default_root_dir=model_dir,
						callbacks=[checkpoint_callback],
						logger=tb_logger,
						enable_progress_bar=False,
						log_every_n_steps=1,
						deterministic=True)

	# Load model if exists, otherwise train it
	if model_path.exists():
		print(f"Dataset \'{dataset_name}\': Model {model_name} with {num_layers=} found. Loading...")
		model = NodeClassificationGNN.load_from_checkpoint(model_path)
	else:
		L.seed_everything(42, workers=True)
		model = NodeClassificationGNN(model_name=model_name, in_channels=dataset.num_node_features,
									  out_channels=dataset.num_classes,
									  num_layers=num_layers, hidden_channels=16, **model_kwargs)
		print(f"Dataset \'{dataset_name}\': Model {model_name} with {num_layers=} not found. Training...")

		trainer.fit(model, data_loader, data_loader)
		model = NodeClassificationGNN.load_from_checkpoint(checkpoint_callback.best_model_path)

	# Print accuracies
	out = trainer.predict(model, data_loader)[0]
	# Training accuracy
	train_acc = accuracy(out[dataset[0].train_mask], dataset[0].y[dataset[0].train_mask], task="multiclass",
						 num_classes=dataset.num_classes)
	# Validation accuracy
	val_acc = accuracy(out[dataset[0].val_mask], dataset[0].y[dataset[0].val_mask], task="multiclass",
					   num_classes=dataset.num_classes)
	# Test accuracy
	test_acc = accuracy(out[dataset[0].test_mask], dataset[0].y[dataset[0].test_mask], task="multiclass",
						num_classes=dataset.num_classes)
	print(f"Train Acc: {100 * train_acc:.3f}%, Val Acc: {100 * val_acc:.3f}%, Test Acc: {100 * test_acc:.3f}%")

	return model, data_loader