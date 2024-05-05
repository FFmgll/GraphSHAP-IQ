from shapiq.explainer.graph.train_gnn_node import get_node_classifier
from shapiq.explainer.graph import GraphSHAPIQ
import numpy as np
from shapiq.games.benchmark.local_xai.benchmark_graph import GraphNodeGame
import torch

from shapiq import ExactComputer

if __name__ == "__main__":

    # Train a GNN for Node Classification
    model, data_loader = get_node_classifier(dataset_name="Cora", model_name="GCN", num_layers=2, bias=True)
    # Evaluate the model
    model.eval()
    torch.no_grad()

    x_graph_full = next(iter(data_loader))  # Take the 1 graph of Cora

    #base_mask = x_graph_full.train_mask
    #base_mask = x_graph.val_mask
    base_mask = torch.zeros(len(x_graph_full.x),dtype=bool)
    base_mask[:12] = True

    x_graph = x_graph_full.subgraph(base_mask)
    class_labels = x_graph.y.numpy()

    game = GraphNodeGame(model=model, max_neighborhood_size=model.n_layers, x_graph=x_graph, class_labels=class_labels)

    explainer = GraphSHAPIQ(game)

    explanation_order = 2


    gt_moebius = {}
    gt_interactions = {}
    node_games = {}
    for node in game._grand_coalition_set:
        node_games[node] = GraphNodeGame(model=model, max_neighborhood_size=model.n_layers, x_graph=x_graph,
                                         class_labels=class_labels, node_id=node)
        exact_computer = ExactComputer(game.n_players, node_games[node])
        gt_interactions[node] = exact_computer.shapley_interaction(index="k-SII", order=explanation_order)
        gt_moebius[node] = exact_computer.moebius_transform()

    gshap_interactions = {}
    gshap_moebius = {}
    sse = {}
    for max_interaction_size in range(1,3):
        gshap_moebius[max_interaction_size], gshap_interactions[max_interaction_size] = explainer.explain(max_interaction_size=max_interaction_size, order=2, efficiency_routine=True)
        sse[max_interaction_size] = {}
        print(max_interaction_size)
        for node in game._grand_coalition_set:
            print(np.sum((gshap_moebius[max_interaction_size][node]-gt_moebius[node]).values**2))


