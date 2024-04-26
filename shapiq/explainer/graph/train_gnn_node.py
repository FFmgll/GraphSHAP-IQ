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

	def __init__(self, model_name: str = "GCN", in_c: int = 1, out_c: int = 1, num_layers: int = 3, bias: bool = False, **kwargs):
		super(NodeClassificationGNN, self).__init__()
		self.save_hyperparameters()
		self.model = getattr(pyg_models, model_name)(in_channels=in_c, # Tested with GCN so far
		                                             hidden_channels=16,
		                                             num_layers=num_layers,
		                                             out_channels=out_c,
		                                             dropout=0.1,
		                                             act="relu",
		                                             bias=bias)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1, weight_decay=2e-3)
		self.criterion = torch.nn.CrossEntropyLoss()

	def forward(self, data, masking="train"):
		"""Forward pass of the model. We have to be extra careful because Cora has only one graph with existing masks."""
		if masking == "train":
			mask = data.train_mask
		elif masking == "val":
			mask = data.val_mask
		elif masking == "test":
			mask = data.test_mask
		elif type(masking) == torch.Tensor:
			mask = masking
		else:
			raise ValueError("In the forward pass, use masking = 'train', 'val', 'test' or provide a boolean tensor to mask nodes")

		out = self.model(data.x, data.edge_index)
		return out[mask], data.y[mask]


	def training_step(self, batch, batch_idx):
		out, y = self.forward(batch, masking="train")
		loss = self.criterion(out, y)
		accuracy = (out.argmax(dim=-1) == y).sum().float().item() / y.sum()
		self.log("train_loss", loss)
		self.log("train_accuracy", accuracy)
		return loss

	def validation_step(self, batch, batch_idx):
		out, y = self.forward(batch, masking="val")
		accuracy = (out.argmax(dim=-1) == y).sum().float().item() / y.sum()
		self.log("val_accuracy", accuracy)

	def test_step(self, batch, batch_idx):
		out, y = self.forward(batch, masking="test")
		accuracy = (out.argmax(dim=-1) == y).sum().float().item() / y.sum()
		self.log("test_accuracy", accuracy)

	def configure_optimizers(self):
		return self.optimizer

def get_node_classifier(dataset_name: str = "Cora", model_name: str = "GCN", num_layers: int = 3, bias: bool = False, **kwargs):
	"""Get the node classifier model, train it if it doesn't exist and return it with the data"""
	dataset = dataset_node_classification(dataset_name)
	data_loader = DataLoader(dataset, batch_size=1)  # There is only 1 graph

	model_path = Path(__file__).resolve().parent / "ckpt" / f"{model_name}_{dataset_name}_{num_layers}_{bias}"

	trainer = L.Trainer(max_epochs=200,
	                    default_root_dir=model_path,
	                    callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor="val_accuracy")],
	                    enable_progress_bar=False,
	                    log_every_n_steps=1)



	#if model_path.exists():
	#	print("Model found. Loading...")
	#	model = NodeClassificationGNN.load_from_checkpoint(model_path)
	#else:
	L.seed_everything(42)
	model = NodeClassificationGNN(model_name=model_name, in_c=dataset.num_node_features, out_c=dataset.num_classes,
	                              num_layers=num_layers, bias=bias, **kwargs)
	print(f"Dataset \'{dataset_name}\': Model {model_name} with {num_layers=} and {bias=} not found. Training...")

	trainer.fit(model, data_loader, data_loader)
	model = NodeClassificationGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

	# Test best model on the test set
	test_result = trainer.test(model, dataloaders=data_loader, verbose=False)
	batch = next(iter(data_loader))
	batch = batch.to(model.device)
	out, y = model.forward(batch, masking="train")
	train_acc = (out.argmax(dim=-1) == y).sum().float().item() / y.sum()
	out, y = model.forward(batch, masking="val")
	val_acc = (out.argmax(dim=-1) == y).sum().float().item() / y.sum()
	result = {"train": train_acc.item(), "val": val_acc.item(), "test": test_result[0]["test_accuracy"]}
	print(result)

	return model, data_loader



if __name__ == "__main__":
	# Train a GNN for Node Classification
	my_beautiful_gnn, data_loader = get_node_classifier(dataset_name="Cora", model_name="GCN", num_layers=2, bias=True)
	# Evaluate the model
	my_beautiful_gnn.eval()
	batch = next(iter(data_loader)) # Take the 1 graph of Cora
	out, y = my_beautiful_gnn.forward(batch, masking="test") # or provide a boolean tensor to mask nodes


