"""This script shows the differences between deep read-out and linear."""
import sys
import warnings

import networkx as nx
import torch_geometric

from shapiq import ExactComputer

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx

from shapiq.explainer.graph import (
    _compute_baseline_value,
    load_graph_model,
)
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.explainer.graph import get_explanation_instances
from shapiq.plot.explanation_graph import explanation_graph_plot
from shapiq.explainer.graph import GraphSHAPIQ


# {39: 11, 293: 11, 410: 10, 438: 7, 524: 9, 648: 7, 935: 9, 1036: 10, 1158: 12, 1255: 10, 1432: 11, 1437: 11, 1446: 10, 1737: 5, 1857: 10, 2012: 11, 2020: 12, 2149: 9, 2358: 10, 2881: 10, 2938: 8, 2997: 7, 3014: 6, 3055: 11, 3285: 10, 3464: 7, 3552: 11, 3616: 11, 4156: 7, 4210: 8}
def print_predictions_players():
    instances = {}
    for i, graph_in in enumerate(get_explanation_instances(DATASET_NAME)):
        num_nodes = graph_in.num_nodes
        og_class = int(graph_in.y.item())
        if num_nodes > 12 or og_class == 1:
            continue
        with torch.no_grad():
            pred_deep = model_deep(graph_in.x, graph_in.edge_index, graph_in.batch)
            pred_linear = model_linear(graph_in.x, graph_in.edge_index, graph_in.batch)

        class_deep = int(torch.argmax(pred_deep).item())
        class_linear = int(torch.argmax(pred_linear).item())
        prob_deep = float(torch.nn.functional.softmax(pred_deep, dim=1)[0, og_class])
        prob_linear = float(torch.nn.functional.softmax(pred_linear, dim=1)[0, og_class])
        if og_class == class_deep == class_linear and abs(prob_deep - prob_linear) < 0.14:
            instances[i] = graph_in.num_nodes
            print(f"{i} {num_nodes}, {prob_deep}, {prob_linear}, {og_class}")
    print(instances)


if __name__ == '__main__':

    DEBUG = False
    if DEBUG:
        warnings.warn("Debug mode is enabled.")

    PLOT_DIR = os.path.join("..", "results", "deep_ro")
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    # model / data parameters
    MODEL_TYPE = "GCN"
    DATASET_NAME = "Mutagenicity"
    N_LAYER = 2
    # 1158 interesting
    DATA_ID = 189
    RANDOM_SEED = 4

    # plot parameter
    COMPACTNESS = 100  # compactness of the graph layout
    SIZE_FACTOR = 2  # factor to scale the node sizes
    N_INTERACTIONS = 100  # number of interactions/explanations to plot for the graph
    CUBIC_SCALING = False  # uses cubic scaling for the node sizes (True) or not (False)
    ADJUST_MIN_MAX = (
        False  # scales the explanation sizes across plots (True) or not (False)
    )
    ADJUST_NODE_POS = (
        False  # adjusts the node positions in the plots (True) or not (False)
    )
    NODE_SIZE_SCALING = 1.0  # scales the node sizes in the plots
    SPRING_K = None  # (None) spring constant for the layout increase for more space between nodes
    INTERACTION_DIRECTION = None  # "positive", "negative", None
    DRAW_THRESHOLD = 0  # threshold for the interaction values to draw the edges

    SAVE_FIG = True
    PLOT_TITLE = False
    INCREASE_FONT_SIZE = True
    if INCREASE_FONT_SIZE:
        plt.rcParams["font.size"] = 16
    APPROXIMATE_SVARMIQ = True
    APPROXIMATE_KERNELSHAPIQ = False
    APPROXIMATION_ORDER = 2

    # get the data point the instance --------------------------------------------------------------
    file_identifier = "_".join([MODEL_TYPE, DATASET_NAME, str(N_LAYER), str(DATA_ID)])
    explanation_instances = get_explanation_instances(DATASET_NAME)
    graph_instance = explanation_instances[DATA_ID]

    # plot the graph -------------------------------------------------------------------------------
    fig, ax = plt.subplots()
    graph = to_networkx(graph_instance, to_undirected=True)
    pos = nx.spring_layout(graph, seed=RANDOM_SEED, k=SPRING_K)
    pos = nx.kamada_kawai_layout(graph, scale=1.0, pos=pos)
    nx.draw(graph, pos, ax=ax, with_labels=True)
    plt.show()

    # get the graph labels -------------------------------------------------------------------------
    graph = to_networkx(graph_instance, to_undirected=True)
    graph_labels, atom_names = None, None
    if DATASET_NAME == "MUTAG":
        atom_names = ["C", "N", "O", "F", "I", "Cl", "Br"]
    if DATASET_NAME == "Mutagenicity":
        atom_names = [
            "C",
            "O",
            "Cl",
            "H",
            "N",
            "F",
            "Br",
            "S",
            "P",
            "I",
            "Na",
            "K",
            "Li",
            "Ca",
        ]
    if DATASET_NAME in ("Benzene", "FluorideCarbonyl", "AlkaneCarbonyl"):
        # link to source: https://github.com/google-research/graph-attribution/blob/main/graph_attribution/featurization.py#L37C1-L41C71
        atom_names = [
            "C",
            "N",
            "O",
            "S",
            "F",
            "P",
            "Cl",
            "Br",
            "Na",
            "Ca",
            "I",
            "B",
            "H",
            "*",
        ]

    if atom_names is not None:
        graph_labels = {
            node_id: atom_names[np.argmax(atom)]
            for node_id, atom in enumerate(graph_instance.x.numpy())
        }

    # load the deep model --------------------------------------------------------------------------
    model_deep = load_graph_model(
        model_type=MODEL_TYPE,
        dataset_name=DATASET_NAME,
        n_layers=N_LAYER,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        jumping_knowledge=False,
        deep_readout=True,
        hidden=128
    )
    model_deep.eval()
    with torch.no_grad():
        prediction_deep = model_deep(
            graph_instance.x, graph_instance.edge_index, graph_instance.batch
        )
    original_class_deep = int(graph_instance.y.item())
    predicted_class_deep = int(torch.argmax(prediction_deep).item())
    predicted_logits_deep = float(prediction_deep[0, predicted_class_deep])
    predicted_prob_deep = float(
        torch.nn.functional.softmax(prediction_deep, dim=1)[0, predicted_class_deep]
    )
    print("Original class: ", original_class_deep)
    print("Predicted class: ", predicted_class_deep)
    print("Predicted logits: ", predicted_logits_deep)
    print("Predicted probability: ", predicted_prob_deep)
    game_deep = GraphGame(
        model_deep,
        x_graph=graph_instance,
        class_id=predicted_class_deep,
        max_neighborhood_size=model_deep.n_layers,
        masking_mode="feature-removal",
        normalize=True,
        baseline=_compute_baseline_value(graph_instance),
    )

    # set up linear model and game -----------------------------------------------------------------
    model_linear = load_graph_model(
        model_type=MODEL_TYPE,
        dataset_name=DATASET_NAME,
        n_layers=N_LAYER,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hidden=128
    )
    model_linear.eval()
    with torch.no_grad():
        prediction_linear = model_linear(
            graph_instance.x, graph_instance.edge_index, graph_instance.batch
        )
    original_class_linear = int(graph_instance.y.item())
    predicted_class_linear = int(torch.argmax(prediction_linear).item())
    predicted_logits_linear = float(prediction_linear[0, predicted_class_linear])
    predicted_prob_linear = float(
        torch.nn.functional.softmax(prediction_linear, dim=1)[0, predicted_class_linear]
    )
    print("Original class: ", original_class_linear)
    print("Predicted class: ", predicted_class_linear)
    print("Predicted logits: ", predicted_logits_linear)
    print("Predicted probability: ", predicted_prob_linear)
    game_linear = GraphGame(
        model_linear,
        x_graph=graph_instance,
        class_id=predicted_class_linear,
        max_neighborhood_size=model_linear.n_layers,
        masking_mode="feature-removal",
        normalize=True,
        baseline=_compute_baseline_value(graph_instance),
    )
    print(f"N Layers {game_linear.max_neighborhood_size}")

    # show_all_predictions -------------------------------------------------------------------------
    print_predictions_players()

    # get receptive field --------------------------------------------------------------------------
    graph_shap_iq = GraphSHAPIQ(game=game_linear)
    neighborhood = graph_shap_iq.neighbors
    valid_interactions, _ = graph_shap_iq._get_all_coalitions(
        max_interaction_size=graph_shap_iq.max_size_neighbors,
        efficiency_routine=False,
    )
    print(neighborhood)
    print("Valid Interactions:", valid_interactions)
    print("N Valid Interactions:", len(valid_interactions))
    print("max_size_neighbors:", graph_shap_iq.max_size_neighbors)

    if DEBUG:
        # manually override valid interactions with 50% powerset
        valid_interactions = []
        from shapiq.utils import powerset
        insert = False
        for interaction in powerset(range(graph_shap_iq.n_players), min_size=1):
            if insert:
                valid_interactions.append(interaction)
                insert = False
            else:
                insert = True
        valid_interactions = set(valid_interactions)
        print("Valid Interactions:", valid_interactions)
        print("N Valid Interactions:", len(valid_interactions))

    # compute möebius ------------------------------------------------------------------------------
    print("Model and Data Loaded")
    print(f"N players: {game_deep.n_players}")
    from shapiq import InteractionValues
    interactions = {}

    try:
        interactions["MI_DEEP"] = InteractionValues.load(f"../results/deep_ro/{file_identifier}_DR_MI.pkl")
        interactions["2-SII_DEEP"] = InteractionValues.load(f"../results/deep_ro/{file_identifier}_DR_2SII.pkl")
        interactions["MI_LINEAR"] = InteractionValues.load(f"../results/deep_ro/{file_identifier}_LR_MI.pkl")
        interactions["2-SII_LINEAR"] = InteractionValues.load(f"../results/deep_ro/{file_identifier}_LR_2SII.pkl")
    except FileNotFoundError:
        print("Computing Deep Explanations")
        computer_deep = ExactComputer(game_fun=game_deep, n_players=game_deep.n_players)
        interactions["MI_DEEP"] = computer_deep(index="Moebius", order=game_deep.n_players)
        interactions["MI_DEEP"].save(f"../results/deep_ro/{file_identifier}_DR_MI.pkl")
        interactions["2-SII_DEEP"] = computer_deep(index="k-SII", order=2)
        interactions["2-SII_DEEP"].save(f"../results/deep_ro/{file_identifier}_DR_2SII.pkl")
        print("Computing Linear Explanations")
        computer_linear = ExactComputer(game_fun=game_linear, n_players=game_linear.n_players)
        interactions["MI_LINEAR"] = computer_linear(index="Moebius", order=game_linear.n_players)
        interactions["MI_LINEAR"].save(f"../results/deep_ro/{file_identifier}_LR_MI.pkl")
        interactions["2-SII_LINEAR"] = computer_linear(index="k-SII", order=2)
        interactions["2-SII_LINEAR"].save(f"../results/deep_ro/{file_identifier}_LR_2SII.pkl")

    # plot the interactions ------------------------------------------------------------------------

    # set the title for the plots ------------------------------------------------------------------
    predicted_prob_deep = round(predicted_prob_deep, 2)
    if predicted_prob_deep == 1.0:
        predicted_prob_deep = "> 0.99"
    elif predicted_prob_deep == 0.0:
        predicted_prob_deep = "< 0.01"
    else:
        predicted_prob_deep = f"= {predicted_prob_deep:.2f}"
    title_suffix_deep = (
        f"Model: {MODEL_TYPE}, Dataset: {DATASET_NAME}, Layers: {N_LAYER}, Data ID: {DATA_ID}, "
        f"Label {original_class_deep}\n"
        rf"Predicted and Explained Class {predicted_class_deep} ($p$ {predicted_prob_deep})"
    )

    # linear model
    predicted_prob_linear = round(predicted_prob_linear, 2)
    if predicted_prob_linear == 1.0:
        predicted_prob_linear = "> 0.99"
    elif predicted_prob_linear == 0.0:
        predicted_prob_linear = "< 0.01"
    else:
        predicted_prob_linear = f"= {predicted_prob_linear:.2f}"
    title_suffix_linear = (
        f"Model: {MODEL_TYPE}, Dataset: {DATASET_NAME}, Layers: {N_LAYER}, Data ID: {DATA_ID}, "
        f"Label {original_class_linear}\n"
        rf"Predicted and Explained Class {predicted_class_linear} ($p$ {predicted_prob_linear})"
    )

    # get min and max values for the color mapping
    min_value = min(
        [
            np.abs(interaction_values.values).min()
            for interaction_values in interactions.values()
        ]
    )
    max_value = max(
        [
            np.abs(interaction_values.values).max()
            for interaction_values in interactions.values()
        ]
    )
    min_max_interactions = None
    if ADJUST_MIN_MAX:
        min_max_interactions = (min_value, max_value)

    # plot möebius interactions --------------------------------------------------------------------
    mi_deep = interactions["MI_DEEP"]
    print(mi_deep)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=mi_deep,
        plot_explanation=True,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
        min_max_interactions=min_max_interactions,
        adjust_node_pos=ADJUST_NODE_POS,
        node_size_scaling=NODE_SIZE_SCALING,
        spring_k=SPRING_K,
        interaction_direction=INTERACTION_DIRECTION,
        draw_threshold=DRAW_THRESHOLD,
    )
    title = "n-SII/MI Explanation (DR)\n" + title_suffix_deep
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_DR_plot_nSII.pdf"))
    plt.show()

    mi_linear = interactions["MI_LINEAR"]
    print(mi_linear)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=mi_linear,
        plot_explanation=True,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
        min_max_interactions=min_max_interactions,
        adjust_node_pos=ADJUST_NODE_POS,
        node_size_scaling=NODE_SIZE_SCALING,
        spring_k=SPRING_K,
        interaction_direction=INTERACTION_DIRECTION,
        draw_threshold=DRAW_THRESHOLD,
    )
    title = "n-SII/MI Explanation (LR)\n" + title_suffix_linear
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_LR_plot_nSII.pdf"))
    plt.show()

    # plot 2-sii -----------------------------------------------------------------------------------

    two_sii_values_deep = interactions["2-SII_DEEP"]
    print(two_sii_values_deep)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=two_sii_values_deep,
        plot_explanation=True,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
        min_max_interactions=min_max_interactions,
        adjust_node_pos=ADJUST_NODE_POS,
        node_size_scaling=NODE_SIZE_SCALING,
        spring_k=SPRING_K,
        interaction_direction=INTERACTION_DIRECTION,
        draw_threshold=DRAW_THRESHOLD,
    )
    title = "2-SII Explanation (DR)\n" + title_suffix_deep
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_DR_plot_2SII.pdf"))
    plt.show()

    two_sii_values_linear = interactions["2-SII_LINEAR"]
    print(two_sii_values_linear)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=two_sii_values_linear,
        plot_explanation=True,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
        min_max_interactions=min_max_interactions,
        adjust_node_pos=ADJUST_NODE_POS,
        node_size_scaling=NODE_SIZE_SCALING,
        spring_k=SPRING_K,
        interaction_direction=INTERACTION_DIRECTION,
        draw_threshold=DRAW_THRESHOLD,
    )
    title = "2-SII Explanation (LR)\n" + title_suffix_linear
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_LR_plot_2SII.pdf"))
    plt.show()

    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=mi_deep,
        plot_explanation=True,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
        min_max_interactions=min_max_interactions,
        adjust_node_pos=ADJUST_NODE_POS,
        node_size_scaling=NODE_SIZE_SCALING,
        spring_k=SPRING_K,
        interaction_direction=INTERACTION_DIRECTION,
        draw_threshold=DRAW_THRESHOLD,
        color_interactions=valid_interactions,
        remove_colored=False
    )
    title = "n-SII/MI Explanation (DR, RF)\n" + title_suffix_deep
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_DR_plot_nSII_receptive_field.pdf"))
    plt.show()

    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=mi_deep,
        plot_explanation=True,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
        min_max_interactions=min_max_interactions,
        adjust_node_pos=ADJUST_NODE_POS,
        node_size_scaling=NODE_SIZE_SCALING,
        spring_k=SPRING_K,
        interaction_direction=INTERACTION_DIRECTION,
        draw_threshold=DRAW_THRESHOLD,
        color_interactions=valid_interactions,
        remove_colored=True
    )
    title = "n-SII/MI Explanation (DR, RF)\n" + title_suffix_deep
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_DR_plot_nSII_receptive_field_removed.pdf"))
    plt.show()

    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=mi_linear,
        plot_explanation=True,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
        min_max_interactions=min_max_interactions,
        adjust_node_pos=ADJUST_NODE_POS,
        node_size_scaling=NODE_SIZE_SCALING,
        spring_k=SPRING_K,
        interaction_direction=INTERACTION_DIRECTION,
        draw_threshold=DRAW_THRESHOLD,
        color_interactions=valid_interactions,
        remove_colored=True
    )
    title = "n-SII/MI Explanation (LR, RF)"
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_LR_plot_nSII_receptive_field_removed.pdf"))
    plt.show()
