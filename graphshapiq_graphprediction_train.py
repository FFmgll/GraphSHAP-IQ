from shapiq.explainer.graph.train_gnn import train_gnn
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Train GNN model for graph prediction')
	parser.add_argument('--dataset_name', type=str, default='', help='Name of the dataset')

	args = parser.parse_args()
	dataset_name = args.dataset_name

	DATASET_NAMES = ["AIDS", "DHFR", "COX2", "BZR", "PROTEINS", "ENZYMES", "MUTAG",
	                 "Mutagenicity"] # TODO: Add more datasets (BBBP, Tox21 from MoleculeNet, Graph-SST, ...)

	# Check if dataset_name is valid, if it is an empty string, or if it is not in the list of valid dataset names
	if dataset_name not in DATASET_NAMES and dataset_name != "":
		raise ValueError("Invalid dataset name. Please choose from the following list: {}".format(DATASET_NAMES))
	else:
		DATASET_NAMES = [dataset_name] if dataset_name != "" else DATASET_NAMES

	# Hyperparameters
	MODEL_TYPES = ["GCN"]  # ["GCN","GIN"]
	N_LAYERS = [3, 1, 2, 4]
	NODE_BIASES = [True]  # [False,True]
	GRAPH_BIASES = [True]  # [False,True]
	HIDDEN = [8, 32, 64, 128]  # 8, 32, 64, 128
	DROPOUT = [True]  # False, True
	BATCH_NORM = [True]  # False, True
	JUMPING_KNOWLEDGE = [True]  # False, True
	DR = False # False, True
	RETRAIN = True  # False, True

	for dataset_name in DATASET_NAMES:
		for model_type in MODEL_TYPES:
			for n_layers in N_LAYERS:
				for node_bias in NODE_BIASES:
					for graph_bias in GRAPH_BIASES:
						for hidden in HIDDEN:
							for dropout in DROPOUT:
								for batch_norm in BATCH_NORM:
									for jumping_knowledge in JUMPING_KNOWLEDGE:
											train_gnn(dataset_name, model_type, n_layers, node_bias, graph_bias, hidden,
											          dropout, batch_norm, jumping_knowledge, DR, RETRAIN)