"""
Here we train a GNN for Node Classification. See below name guard for the main function and usage.
Datasets in order of complexity:
- Cora
TODO:
- Citeseer
- Pubmed
(other possible datasets: Chameleon, Squirrel, Actor, Coauthor, CS, Physics)
For node classification on transductive datasets, we will use the following:
BA-Shapes (adapted to transductive setting)
Models:
- GCN
- GAT
- GIN

Besides, later I will set up a hyperparameter search for the best model and dataset combination :-)
"""
import torch
import torch_geometric

torch.set_float32_matmul_precision('high')

import logging

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

# import the NodeClassificationGNN class
# import the dataset_node_classification function
# import the get_node_classifier function
from shapiq.explainer.graph.graph_models.node_prediction import get_node_classifier

if __name__ == "__main__":
	# Train a GNN for Node Classification
	# How to use: just import the function get_node_classifier and use it with the desired parameters
	# my_beautiful_gnn, dataset, data_loader = get_node_classifier(dataset_name="Cora", model_name="GCN", num_layers=2)

	# Here we train some models for Node Classification and store them in the ckpt/node_prediction directory
	def train_sweep():
		print("Starting training sweep...")
		DATASET_NAMES = ["BAShapes"]
		MODEL_TYPES = ["GCN"]
		N_LAYERS = [3]

		for dataset_name in DATASET_NAMES:
			for model_type in MODEL_TYPES:
				for n_layers in N_LAYERS:
					my_beautiful_gnn, dataset, data_loaders = get_node_classifier(dataset_name=dataset_name,
																				 model_name=model_type,
																				 num_layers=n_layers)


	train_sweep()


	def experiments_on_masking_1():
		"I want to test if the logits predictions are the same when masking the graph with 0 outside a 2-hop neighbourhood"
		my_beautiful_gnn, data_loader = get_node_classifier(dataset_name="Cora", model_name="GCN", num_layers=2)
		my_beautiful_gnn.eval()
		with torch.no_grad():
			# Get the baseline values
			baseline_value = my_beautiful_gnn(data_loader.dataset[0])
			# Get only the correct predictions
			out = my_beautiful_gnn(data_loader.dataset[0])
			pred = out.argmax(dim=1)
			correct_samples_index = pred == data_loader.dataset[0].y
			# Get the indexes of the correct samples
			correct_samples_index = torch.where(correct_samples_index)[0]
			# Loop over the correct samples
			identical = 0
			counter = 0
			for node in correct_samples_index:
				# Print a counter
				print(f"Node {counter}/{len(correct_samples_index)}", end="\r")
				counter += 1
				# Get the 2-hop neighbourhood
				subset, edge_index, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(node_idx=node.item(),
																							  num_hops=2,
																							  edge_index=
																							  data_loader.dataset[
																								  0].edge_index)
				# Set to zero all the nodes features that are not in subset
				data_masked = data_loader.dataset[0].clone()
				set_to_zero = ~torch.isin(torch.arange(data_masked.num_nodes), subset)
				data_masked.x[set_to_zero] = 0
				# Compute the prediction with the masked graph
				pred_masked = my_beautiful_gnn(data_masked)
				# Compare the prediction with the baseline value for the node
				node_baseline = baseline_value[node]
				node_masked = pred_masked[node]
				if torch.allclose(node_baseline, node_masked):
					identical += 1
			# Finally
			print(f"Identical predictions: {identical}/{len(correct_samples_index)}")  # It works!


	@torch.no_grad()
	def experiments_on_masking_2():
		"I want to test if the logits predictions are the same when taking a subgraph of the graph"
		my_beautiful_gnn, data_loader = get_node_classifier(dataset_name="Cora", model_name="GCN", num_layers=2)
		my_beautiful_gnn.eval()
		# Get the baseline values
		baseline_value = my_beautiful_gnn(data_loader.dataset[0])
		# Get only the correct predictions
		out = my_beautiful_gnn(data_loader.dataset[0])
		pred = out.argmax(dim=1)
		correct_samples_index = pred == data_loader.dataset[0].y
		# Get the indexes of the correct samples
		correct_samples_index = torch.where(correct_samples_index)[0]
		# Loop over the correct samples
		identical = 0
		counter = 0
		for node in correct_samples_index:
			# Print a counter
			print(f"Node {counter}/{len(correct_samples_index)}", end="\r")
			counter += 1
			# Get the subgraph induced by the 2-hop neighbourhood
			subset, edge_index, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(node_idx=node.item(),
																						  num_hops=2,
																						  edge_index=
																						  data_loader.dataset[
																							  0].edge_index,
																						  relabel_nodes=True)  # THIS IS IMPORTANT

			# Get the subgraph
			data_subgraph = data_loader.dataset[0].clone()
			data_subgraph.x = data_subgraph.x[subset]
			data_subgraph.y = data_subgraph.y[subset]
			data_subgraph.edge_index = edge_index
			# Compute the prediction with the masked graph
			pred_subgraph = my_beautiful_gnn(data_subgraph)
			# Compare the prediction with the baseline value for the node
			node_baseline = baseline_value[node]
			node_subgraph = pred_subgraph[mapping]
			if torch.allclose(node_baseline, node_subgraph, atol=0.3):  # High tolerance!
				identical += 1
		# Finally
		print(f"Identical predictions: {identical}/{len(correct_samples_index)}")  # ... not so good :-(