"""This method trains the GNN architectures and stores them under /ckpt.
Naming convention is MODELTYPE_DATASET_NLAYERS_NODEBIAS_GRAPHBIAS, e.g. GCN_MUTAG_3_False_False.
The corresponding directory is MODELTYPE/DATASET, e.g. GCN/MUTAG"""


import os
import torch
from torch_geometric.loader import DataLoader
from graphxai_local.datasets import (
    MUTAG,
)  # renamed to avoid conflict with potential installs

from shapiq.explainer.graph.graph_models import GCN, GIN

from shapiq.explainer.graph.graph_datasets import CustomTUDataset

def get_TU_dataset(device,name):
    # Load dataset
    dataset = CustomTUDataset(root="shapiq/explainer/graph/graph_datasets", name=name, seed=1234, split_sizes=(0.8, 0.1, 0.1))
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


def train_and_store(model,train_loader, val_loader, test_loader, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Train and test functions
    def train(graph_model):
        graph_model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
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
            out = graph_model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    # set to True to train the model or False to load the best model from the checkpoint
    TRAIN = True

    if TRAIN:
        best_val_acc = 0
        for epoch in range(1, 100):
            train(graph_model=model)  # uncomment to train
            train_acc = test(train_loader, graph_model=model)
            val_acc = test(val_loader, graph_model=model)
            test_acc = test(test_loader, graph_model=model)

            # Save best model on validation set
            if epoch == 1 or val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                #print(f"Best model saved at epoch {epoch}")
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


def train_gnn(dataset_name,model_type,n_layers, node_bias, graph_bias,enforce_retrain=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_name in ["AIDS","DHFR","COX2","BZR","MUTAG","BENZENE","PROTEINS","ENZYMES","Mutagenicity"]:
        train_loader, val_loader, test_loader, num_nodes_features, num_classes = get_TU_dataset(device, dataset_name)

    if model_type == "GCN":
        model = GCN(in_channels=num_nodes_features, hidden_channels=64, out_channels=num_classes,n_layers=n_layers,graph_bias=graph_bias, node_bias=node_bias).to(device)
    if model_type == "GIN":
        model = GIN(in_channels=num_nodes_features, hidden_channels=64, out_channels=num_classes, n_layers=n_layers,
                    graph_bias=graph_bias, node_bias=node_bias).to(device)

    model_id = "_".join([model_type,dataset_name,str(n_layers),str(node_bias),str(graph_bias)])
    # Construct the path to the target directory
    target_dir = os.path.join("shapiq","explainer","graph","ckpt", "graph_prediction",model_type,dataset_name)

    save_path = os.path.join(target_dir,model_id+".pth")


    # Check if the directory exists, if not, create it
    if not os.path.exists(target_dir):
        raise Exception("Please create directory",target_dir)


    if enforce_retrain:
        print("Training model ", model_id)
        train_and_store(model,train_loader,val_loader,test_loader,save_path)
    else:
        if os.path.exists(save_path):
            print("Loading model ", model_id)
        else:
            # Train model
            print("Training model ", model_id)
            train_and_store(model, train_loader, val_loader, test_loader, save_path)

    model.load_state_dict(torch.load(save_path))
    return model, model_id
