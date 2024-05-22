"""This script conducts an ablation study on a model that has nonlinear output (MLP) layers by
computing a graph game for small graphs (<12 nodes) with the exact computer and with GraphSHAP-IQ
and comparing the results."""
import numpy as np

from shapiq.explainer.graph import GraphSHAPIQ
from shapiq.explainer.graph.utils import (
    load_graph_model,
    get_explanation_instances,
    _compute_baseline_value,
)
from shapiq.games.benchmark.local_xai.benchmark_graph import GraphGame
from shapiq.exact import ExactComputer


def run_ablation(deep_model, data_point) -> tuple[float, float]:
    # make it a game -------------------------------------------------------------------------------
    baseline_value = _compute_baseline_value(data_point)
    game = GraphGame(
        model=deep_model,
        x_graph=data_point,
        max_neighborhood_size=N_LAYER,
        baseline=baseline_value,
    )

    # compute values with GraphSHAPIQ --------------------------------------------------------------
    print("Computing GraphSHAPIQ values...")
    explainer = GraphSHAPIQ(game)
    if explainer.total_budget > 20_000:
        # skip
        print(f"Skipping due to high budget {explainer.total_budget}")
        return 0.0, 0.0
    explainer_values, _ = explainer.explain(order=game.n_players)

    # compute the exact game -----------------------------------------------------------------------
    print("Computing exact values...")
    computer = ExactComputer(game.n_players, game)
    exact_values = computer(index="Moebius", order=game.n_players)

    # compare the results --------------------------------------------------------------------------
    differences = exact_values - explainer_values
    differences_sum = sum(np.abs(differences.values))
    print("Exact values:", exact_values)
    print("GraphSHAPIQ values:", explainer_values)
    print("Difference:", differences)
    print("Sum of diff:", differences_sum)

    diff_manual = 0.0
    for interaction in exact_values.interaction_lookup.keys():
        diff_manual += np.abs(exact_values[interaction] - explainer_values[interaction])
    for interaction in explainer_values.interaction_lookup.keys():
        if interaction not in exact_values.interaction_lookup:
            diff_manual += np.abs(exact_values[interaction] - explainer_values[interaction])
    print("Sum of diff (manual):", diff_manual)

    return differences_sum, diff_manual


if __name__ == "__main__":

    # model parameters -----------------------------------------------------------------------------
    DATASET_NAME = "PROTEINS"
    MODEL_TYPE = "GIN"
    N_LAYER = 2

    # run parameters -------------------------------------------------------------------------------
    N_RUNS = 15
    MAX_NODES = 12

    # get the model --------------------------------------------------------------------------------
    model = load_graph_model(
        dataset_name=DATASET_NAME,
        model_type=MODEL_TYPE,
        n_layers=N_LAYER,
        deep_readout=True,
        jumping_knowledge=False,
    )
    print(model)

    # get the explanation instances ----------------------------------------------------------------
    explanation_instances = get_explanation_instances(DATASET_NAME)
    # select only instances with less than MAX_NODES nodes
    small_nodes = []
    for graph in explanation_instances:
        if graph.num_nodes <= MAX_NODES:
            small_nodes.append(graph)
    print(f"Number of instances with less than {MAX_NODES} nodes:", len(small_nodes))

    diffs, diffs_manual = 0, 0
    for i in range(N_RUNS):
        print(f"Run {i+1}/{N_RUNS}")
        diff = run_ablation(model, small_nodes[i])
        diffs += diff[0]
        diffs_manual += diff[1]
        print("\n")

    print(f"Average difference: {diffs/N_RUNS}")
    print(f"Average difference (manual): {diffs_manual/N_RUNS}")
