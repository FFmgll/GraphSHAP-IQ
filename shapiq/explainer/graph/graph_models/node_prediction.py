from pathlib import Path
import json

import torch
import torch_geometric.nn.models as pyg_models

from torch_geometric.datasets import Planetoid, ExplainerDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets.graph_generator import BAGraph
import torch_geometric.transforms as T

import lightning as L
from torchmetrics.classification import Accuracy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


# Load dataset
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

	def __init__(self, model_name: str = "GCN", cfg: dict = None):
		super().__init__()
		self.save_hyperparameters(ignore="cfg")
		self.model = getattr(pyg_models, model_name)(**cfg["model_kwargs"])
		self.optimizer = torch.optim.Adam(self.model.parameters(),
										  lr=cfg["learning_rate"],
										  weight_decay=cfg["weight_decay"])
		wieghts = torch.tensor(cfg["class_weights"], dtype=torch.float32) if "class_weights" in cfg else None
		self.loss = torch.nn.CrossEntropyLoss(weight=wieghts)
		out_dim = cfg["model_kwargs"]["out_channels"]
		task = "multiclass" if out_dim > 1 else "binary"
		self.train_acc = Accuracy(task=task, num_classes=out_dim)
		self.val_acc = Accuracy(task=task, num_classes=out_dim)
		self.test_acc = Accuracy(task=task, num_classes=out_dim)

	def forward(self, data):
		"""Forward pass of the model. Returns the logits."""
		out = self.model(data.x, data.edge_index)
		return out

	def training_step(self, batch, batch_idx):
		out = self.forward(batch)
		if hasattr(batch, "train_mask"):
			mask = batch.train_mask  # Get the mask for the training nodes
			loss = self.loss(out[mask], batch.y[mask])
			self.train_acc(out[mask], batch.y[mask])
		else:
			loss = self.loss(out, batch.y)
			self.train_acc(out, batch.y)

		self.log("train_loss", loss, batch_size=batch.size(0))
		self.log("train_accuracy", self.train_acc, batch_size=batch.size(0), prog_bar=True, on_step=False,
		         on_epoch=True)
		return loss

	def validation_step(self, batch, batch_idx):
		out = self.forward(batch)
		if hasattr(batch, "val_mask"):
			mask = batch.val_mask
			self.val_acc.update(out[mask], batch.y[mask])
		else:
			self.val_acc.update(out, batch.y)

		self.log("val_accuracy", self.val_acc, batch_size=batch.size(0), prog_bar=True, on_step=False, on_epoch=True)

	def test_step(self, batch, batch_idx):
		out = self.forward(batch)
		if hasattr(batch, "test_mask"):
			mask = batch.test_mask
			self.test_acc.update(out[mask], batch.y[mask])
		else:
			self.test_acc(out, batch.y)

		self.log("test_accuracy", self.train_acc, batch_size=batch.size(0), prog_bar=False, on_step=False,
		         on_epoch=True)

	def configure_optimizers(self):
		return self.optimizer


def get_node_classifier(dataset_name: str = "Cora", model_name: str = "GCN", num_layers: int = 3,
						**model_kwargs):
	"""Get the node classifier model, train it if it doesn't exist and return it with the data"""

	def dataset_node_classification(dataset_name: str = "Cora"):
		if dataset_name == "Cora":
			"This loads the dataset for node classification with the standard splits"
			dataset_path = Path(__file__).resolve().parents[4] / "data"  # Path to the data folder in main directory
			dataset = Planetoid(root=dataset_path, name=dataset_name)
			return dataset
		if dataset_name == "BAShapes":
			L.seed_everything(2024, workers=True)
			dataset = ExplainerDataset(graph_generator=BAGraph(num_nodes=10, num_edges=8),
									   motif_generator='house',
									   num_motifs=2,
									   num_graphs=500,
									   transform=T.Constant())
			return dataset

	# Hyperparameters
	cfg = json.load(open(Path(__file__).parent.resolve() / "config.json"))
	cfg = cfg["node"][dataset_name]
	cfg[model_name]["model_kwargs"].update(model_kwargs)
	cfg[model_name]["model_kwargs"]["num_layers"] = num_layers

	# Data
	dataset = dataset_node_classification(dataset_name)
	if dataset_name == "Cora":
		# Cora dataset has a fixed split
		data_loader = DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=4, persistent_workers=True)
		train_loader = data_loader
		val_loader = data_loader
		test_loader = data_loader
	else:
		# Split the dataset in 70% train, 15% validation and 15% test
		train_loader = DataLoader(dataset[:int(len(dataset) * 0.7)], batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
		val_loader = DataLoader(dataset[int(len(dataset) * 0.7):int(len(dataset) * 0.85)], batch_size=cfg["batch_size"],
								shuffle=False, num_workers=4)
		test_loader = DataLoader(dataset[int(len(dataset) * 0.85):], batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

	cfg[model_name]["model_kwargs"]["in_channels"] = dataset.num_node_features
	cfg[model_name]["model_kwargs"]["out_channels"] = dataset.num_classes

	# Model
	model_dir = Path(__file__).resolve().parents[1] / "ckpt" / "node_prediction" / f"{model_name}" / f"{dataset_name}"
	model_path = model_dir / f"{model_name}_{dataset_name}_{num_layers}.pth"

	# Load model if exists, otherwise train it
	if model_path.exists():
		print(f"Dataset \'{dataset_name}\': Model {model_name} with {num_layers=} found. Loading...")
		model = NodeClassificationGNN.load_from_checkpoint(model_path, cfg=cfg[model_name])
	else:
		L.seed_everything(42, workers=True)

		# Trainer and callbacks
		checkpoint_callback = ModelCheckpoint(save_weights_only=True,
											  mode='max',
											  monitor="val_accuracy",
											  dirpath=model_dir,
											  filename=f"{model_name}_{dataset_name}_{num_layers}")
		checkpoint_callback.FILE_EXTENSION = ".pth"

		tb_logger = TensorBoardLogger(save_dir=Path(__file__).resolve().parents[1] / "ckpt" / "training_logs",
									  default_hp_metric=True)
		tb_logger.log_hyperparams(params={"dataset": dataset_name,
										  "model": model_name,
										  **cfg[model_name]["model_kwargs"]},
								  metrics={"hp/validated_train_accuracy": 0,
										   "hp/validated_val_accuracy": 0,
										   "hp/validated_test_accuracy": 0})

		trainer = L.Trainer(max_epochs=cfg[model_name]["max_epochs"],
							default_root_dir=model_dir,
							callbacks=[checkpoint_callback],
							logger=tb_logger,
							enable_progress_bar=True,
							log_every_n_steps=1,
							deterministic=True)

		model = NodeClassificationGNN(model_name=model_name, cfg=cfg[model_name])
		print(f"Dataset \'{dataset_name}\': Model {model_name} with {num_layers=} not found. Training...")

		trainer.fit(model, train_loader, val_loader)
		model = NodeClassificationGNN.load_from_checkpoint(checkpoint_callback.best_model_path, cfg=cfg[model_name])

		# Print accuracies
		out = trainer.predict(model, test_loader)[0]
		if dataset_name == "Cora": # THIS IS CRAP
			# Training accuracy computed as comparison of the logits with the true labels
			train_acc = 100 * (
						(out[dataset[0].train_mask].argmax(dim=-1) == dataset[0].y[dataset[0].train_mask]).sum().item() /
						dataset[0].train_mask.sum().item())
			# Validation accuracy
			val_acc = 100 * ((out[dataset[0].val_mask].argmax(dim=-1) == dataset[0].y[dataset[0].val_mask]).sum().item() /
							 dataset[0].val_mask.sum().item())
			# Test accuracy
			test_acc = 100 * (
						(out[dataset[0].test_mask].argmax(dim=-1) == dataset[0].y[dataset[0].test_mask]).sum().item() /
						dataset[0].test_mask.sum().item())

			# Log the metrics
			print(f"Train Acc: {train_acc:.3f}%, Val Acc: {val_acc:.3f}%, Test Acc: {test_acc:.3f}%")
			tb_logger.log_metrics({"hp/validated_train_accuracy": train_acc, "hp/validated_val_accuracy": val_acc,
								   "hp/validated_test_accuracy": test_acc})
		else:
			# Compute the accuracy on the test dataloader
			trainer.test(model, test_loader, verbose=True)

	return model, dataset, [train_loader, val_loader, test_loader]