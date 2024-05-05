from shapiq.explainer.graph.train_gnn import train_gnn


if __name__ == "__main__":
    DATASET_NAMES =  ["AIDS","DHFR","COX2","BZR","PROTEINS", "ENZYMES", "MUTAG", "Mutagenicity"] #["AIDS","DHFR","COX2","BZR","PROTEINS", "ENZYMES", "MUTAG", "Mutagenicity"]
    MODEL_TYPES = ["GCN"]  # ["GCN","GIN"]
    N_LAYERS = [1, 2, 3, 4]
    NODE_BIASES = [True]  # [False,True]
    GRAPH_BIASES = [True]  # [False,True]
    RETRAIN = False # False, True

    for dataset_name in DATASET_NAMES:
        for model_type in MODEL_TYPES:
            for n_layers in N_LAYERS:
                for node_bias in NODE_BIASES:
                    for graph_bias in GRAPH_BIASES:
                        train_gnn(dataset_name=dataset_name, model_type=model_type, n_layers=n_layers, node_bias=node_bias,
                                  graph_bias=graph_bias, enforce_retrain=RETRAIN)
