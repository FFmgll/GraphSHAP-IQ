import pandas as pd

from shapiq.explainer.graph import GraphSHAPIQ
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.explainer.graph import get_explanation_instances

import torch
import numpy as np
import os



def dummy_model(x,edge_index,batch):
    # return 20 dummy classes
    return torch.zeros((x.shape[0],20))

def dummy_eval():
    return None

def evaluate_complexity(
    dataset_name, n_layers, all_samples_to_explain, masking_mode="feature-removal"
):
    gshap_budget = {}
    gshap_budget_est = {}
    players = {}
    max_neighborhood_size = {}

    for data_id, x_graph in enumerate(all_samples_to_explain):
        dummy_module = torch.nn.Module()
        dummy_module.__call__ = dummy_model
        dummy_module.eval = dummy_eval
        game = GraphGame(
            dummy_module,
            x_graph=x_graph,
            class_id=x_graph.y.item(),
            max_neighborhood_size=n_layers,
            masking_mode=masking_mode,
            normalize=False,
            baseline=0,
        )
        # setup the explainer
        gSHAP = GraphSHAPIQ(game)

        gshap_budget[data_id] = gSHAP.total_budget
        players[data_id] = gSHAP.n_players
        max_neighborhood_size[data_id] = gSHAP.max_neighborhood_size
        gshap_budget_est[data_id] = gSHAP.budget_estimated

    results = pd.DataFrame(
        index=gshap_budget.keys(), data=gshap_budget.values(), columns=["budget"]
    )
    results["budget_upper_bound"] = gshap_budget_est
    results["max_neighborhood_size"] = max_neighborhood_size
    results["n_players"] = players

    save_name = "_".join(["complexity", dataset_name + "_" + str(n_layers)])
    save_path = os.path.join("../results/complexity_analysis", save_name + ".csv")
    results.to_csv(save_path)


if __name__ == "__main__":
    DATASET_NAMES = [
        #"AIDS",
        #"DHFR",
        #"COX2",
        #"BZR",
        #"PROTEINS",
        #"ENZYMES",
        #"MUTAG",
        #"Mutagenicity",
        'FluorideCarbonyl',
        'Benzene',
        'AlkaneCarbonyl'
    ]  # ["AIDS","DHFR","COX2","BZR","PROTEINS", "ENZYMES", "MUTAG", "Mutagenicity"]
    N_LAYERS = [1, 2, 3, 4]

    for dataset_name in DATASET_NAMES:
        all_samples_to_explain = get_explanation_instances(dataset_name)
        for n_layers in N_LAYERS:
            evaluate_complexity(dataset_name, n_layers, all_samples_to_explain)
