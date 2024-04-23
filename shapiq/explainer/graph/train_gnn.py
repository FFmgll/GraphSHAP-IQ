from pathlib import Path
import os
import torch
from torch_geometric.loader import DataLoader
from graphxai_local.datasets import (
    MUTAG,
)  # renamed to avoid conflict with potential installs

from graphxai_local.gnn_models.graph_classification import GCN_3layer


def train_gnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
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

    # Model and optimizer
    model_list = {}
    model_list["GCN3"] = GCN_3layer

    for CURRENT_MODEL_NAME, CURRENT_MODEL in model_list.items():
        model = CURRENT_MODEL(num_nodes_features, 64, num_classes).to(device)

        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the target directory
        target_dir = os.path.normpath(
            os.path.join(current_dir, "../../../graphxai_local/gnn_models/ckpt")
        )

        # Check if the directory exists, if not, create it
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        save_path = Path(__file__).resolve().parent.parent.parent.parent / "graphxai_local" / "gnn_models" / "ckpt" / (CURRENT_MODEL_NAME+"_test.pth")

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
                print(
                    f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
                )

                # Save best model on validation set
                if epoch == 1 or val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), save_path)
                    print(f"Best model saved at epoch {epoch}")
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
