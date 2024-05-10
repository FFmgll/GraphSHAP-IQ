"""This script is used to verify that GraphSHAP-IQ produces the exact same values as the
ExactComputer for a given instance."""

from shapiq.explainer.graph import GraphSHAPIQ, get_explanation_instances, load_graph_model
from shapiq.explainer.graph.utils import _compute_baseline_value
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.exact import ExactComputer


if __name__ == "__main__":

    DATASET_NAME = "Mutagenicity"
    MODEL_ID = "GCN"
    N_LAYERS = 1

    N_PLAYERS = 12

    # setup the game
    model = load_graph_model(model_type=MODEL_ID, dataset_name=DATASET_NAME, n_layers=N_LAYERS)
    model.eval()
    explanation_instances = get_explanation_instances(DATASET_NAME)
    games = []
    for data_id, x_graph in enumerate(explanation_instances):
        baseline = _compute_baseline_value(x_graph)
        game_to_run = GraphGame(
            model,
            x_graph=x_graph,
            class_id=x_graph.y.item(),
            max_neighborhood_size=model.n_layers,
            masking_mode="feature-removal",
            normalize=True,
            baseline=baseline,
            instance_id=int(data_id),
        )
        if game_to_run.n_players == N_PLAYERS:
            games.append(game_to_run)
            break

    game = games[0]

    # run the exact computation
    print("Running the exact computation.")
    exact_computer = ExactComputer(n_players=game.n_players, game_fun=game)
    exact_values = exact_computer(index="Moebius", order=game.n_players)
    print("Exact values:")
    print(exact_values)

    # run the GraphSHAP-IQ algorithm
    print("Running the GraphSHAP-IQ algorithm.")
    graph_shapiq = GraphSHAPIQ(game)
    graph_shapiq_values, _ = graph_shapiq.explain(
        max_interaction_size=graph_shapiq.max_size_neighbors, order=game.n_players
    )
    print("GraphSHAP-IQ values:")
    print(graph_shapiq_values)

    # print setup from graphshapiq
    print("GraphSHAP-IQ setup:")
    print("n_players: ", graph_shapiq.n_players)
    print("max_size_neighbors: ", graph_shapiq.max_size_neighbors)

    # verify that the values are the same
    for interaction in exact_values.interaction_lookup:
        # test if exact values are approximately the same as GraphSHAP-IQ values
        tolerance = 1e-4
        if abs(exact_values[interaction] - graph_shapiq_values[interaction]) > tolerance:
            raise ValueError(
                f"Values differ for interaction {interaction} with values "
                f"{exact_values[interaction]} and {graph_shapiq_values[interaction]}."
            )
