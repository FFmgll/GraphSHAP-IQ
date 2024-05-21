"""This script is used to visualize the explanation graphs for the GraphSHAP-IQ approximation."""
import copy
import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx

from shapiq.explainer.graph import _compute_baseline_value, GraphSHAPIQ, load_graph_model
from shapiq.games.benchmark.local_xai import GraphGame
from shapiq.explainer.graph import get_explanation_instances
from shapiq.interaction_values import InteractionValues
from shapiq.plot.explanation_graph import explanation_graph_plot
from shapiq.moebius_converter import MoebiusConverter

PLOT_DIR = os.path.join("..", "results", "explanation_graphs")
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


if __name__ == "__main__":

    # cool insights from the explanation
    # GCN, Mutagenicity, 2, 189, (RANDOM_SEED = 4)  - NO2 group interaction
    # GCN, Mutagenicity, 2, 71, (RANDOM_SEED = 10)  - NO2 group interaction

    INDEX = "k-SII"

    MODEL_TYPE = "GAT"
    DATASET_NAME = "Benzene"
    N_LAYER = 3
    DATA_ID = 57

    GET_PYRIDINE = False  # get a Pyridine molecule

    # plot parameter
    RANDOM_SEED = 1  # random seed for the graph layout
    COMPACTNESS = 100  # compactness of the graph layout
    SIZE_FACTOR = 2  # factor to scale the node sizes
    N_INTERACTIONS = 100  # number of interactions/explanations to plot for the graph
    CUBIC_SCALING = False  # uses cubic scaling for the node sizes (True) or not (False)
    ADJUST_MIN_MAX = False  # scales the explanation sizes across plots (True) or not (False)
    ADJUST_NODE_POS = False  # adjusts the node positions in the plots (True) or not (False)
    NODE_SIZE_SCALING = 1.0  # scales the node sizes in the plots
    SPRING_K = None  # (None) spring constant for the layout increase for more space between nodes
    INTERACTION_DIRECTION = None  # "positive", "negative", None
    DRAW_THRESHOLD = 0.0  # threshold for the interaction values to draw the edges

    RUN_MODEL = False
    SAVE_FIG = True
    PLOT_TITLE = False
    INCREASE_FONT_SIZE = True

    # for saving the plots
    file_identifier = "_".join([MODEL_TYPE, DATASET_NAME, str(N_LAYER), str(DATA_ID)])
    if GET_PYRIDINE:
        file_identifier = "_".join([MODEL_TYPE, DATASET_NAME, str(N_LAYER), "Pyridine"])

    # increase font size for the plots and set bold
    if INCREASE_FONT_SIZE:
        plt.rcParams["font.size"] = 16
        plt.rcParams["font.weight"] = "bold"

    # get the data point the instance --------------------------------------------------------------

    if GET_PYRIDINE:  # get a benzene ring with a Nitrogen atom in the ring
        DATASET_NAME = "Benzene"
        DATA_ID = 42
        explanation_instances = get_explanation_instances(DATASET_NAME)
        graph_instance = explanation_instances[DATA_ID]
        DATA_ID = "Pyridine"
        binary_mask = torch.zeros_like(graph_instance.x[:, 0], dtype=torch.bool)
        binary_mask[7] = True
        binary_mask[8] = True
        binary_mask[9] = True
        binary_mask[10] = True
        binary_mask[11] = True
        binary_mask[12] = True
        # apply the mask
        graph_instance = graph_instance.subgraph(binary_mask)

        # plot the graph
        fig, ax = plt.subplots()
        graph = to_networkx(graph_instance, to_undirected=True)
        pos = nx.spring_layout(graph, seed=RANDOM_SEED)
        nx.draw(graph, pos, ax=ax, with_labels=True, node_size=1000, node_color="lightblue")
        ax.set_title("Original Graph")
        plt.show()
    else:
        explanation_instances = get_explanation_instances(DATASET_NAME)
        graph_instance = explanation_instances[DATA_ID]

    # load the model and get the prediction --------------------------------------------------------
    model = load_graph_model(
        model_type=MODEL_TYPE,
        dataset_name=DATASET_NAME,
        n_layers=N_LAYER,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model.eval()
    with torch.no_grad():
        prediction = model(graph_instance.x, graph_instance.edge_index, graph_instance.batch)
    original_class = int(graph_instance.y.item())
    predicted_class = int(torch.argmax(prediction).item())
    predicted_logits = float(prediction[0, predicted_class])
    predicted_prob = float(torch.nn.functional.softmax(prediction, dim=1)[0, predicted_class])
    print("Original class: ", original_class)
    print("Predicted class: ", predicted_class)
    print("Predicted logits: ", predicted_logits)
    print("Predicted probability: ", predicted_prob)
    game = GraphGame(
        model,
        x_graph=graph_instance,
        class_id=predicted_class,
        max_neighborhood_size=model.n_layers,
        masking_mode="feature-removal",
        normalize=True,
        baseline=_compute_baseline_value(graph_instance),
    )

    # run the model on the dataset and get the prediction ------------------------------------------
    if RUN_MODEL and DATASET_NAME == "Mutagenicity":
        results = []
        for i, instance in enumerate(explanation_instances):
            label = int(instance.y.item())
            with torch.no_grad():
                prediction = model(instance.x, instance.edge_index, instance.batch)
            predicted_class = int(torch.argmax(prediction))
            predicted_logits = float(prediction[0, predicted_class])
            predicted_prob = float(
                torch.nn.functional.softmax(prediction, dim=1)[0, predicted_class]
            )
            result = {
                "data_id": i,
                "label": label,
                "n_nodes": instance.num_nodes,
                "predicted_class": predicted_class,
                "predicted_logits": predicted_logits,
                "predicted_prob": predicted_prob,
            }
            results.append(result)
        df = pd.DataFrame(results)
        correct = df["label"] == df["predicted_class"]
        df["correct"] = correct
        df = df[(df["correct"]) & (df["label"] == 0)]
        df = df.sort_values(by="n_nodes", ascending=True)
        df.to_csv(os.path.join(PLOT_DIR, "mutagenicity_results.csv"), index=False)

    # set the title for the plots ------------------------------------------------------------------

    baseline_value = game.normalization_value
    predicted_prob = round(predicted_prob, 2)
    if predicted_prob == 1.0:
        predicted_prob = "> 0.99"
    elif predicted_prob == 0.0:
        predicted_prob = "< 0.01"
    else:
        predicted_prob = f"= {predicted_prob:.2f}"

    # set the title suffix for the plots
    title_suffix = (
        f"Model: {MODEL_TYPE}, Dataset: {DATASET_NAME}, Layers: {N_LAYER}, Data ID: {DATA_ID}, "
        f"Label {original_class}\n"
        rf"Predicted and Explained Class {predicted_class} ($p$ {predicted_prob}, "
        f"logits. = {predicted_logits:.2f})"
    )

    # get explanation values -----------------------------------------------------------------------

    print(f"Running the explanation...")
    explainer = GraphSHAPIQ(game=game, verbose=True)
    moebius_values, _ = explainer.explain()
    if SAVE_FIG:
        moebius_values.save(os.path.join(PLOT_DIR, f"{file_identifier}.interaction_values"))

    # create the interaction values to plot --------------------------------------------------------

    converter = MoebiusConverter(moebius_coefficients=moebius_values)
    interactions: dict[str, InteractionValues] = {
        "n-SII": copy.deepcopy(moebius_values),
        "6-SII": converter(index=INDEX, order=6),
        "3-SII": converter(index=INDEX, order=3),
        "2-SII": converter(index=INDEX, order=2),
        "SV": converter(index=INDEX, order=1),
        "2-STII": converter(index="STII", order=2),
        "6-STII": converter(index="STII", order=6),
        "SII-2": converter(index="SII", order=2),
        "SII-6": converter(index="SII", order=6),
        "2-FSII": converter(index="FSII", order=2),
        "6-FSII": converter(index="FSII", order=6),
    }
    # get min and max values for the color mapping
    min_value = min(
        [np.abs(interaction_values.values).min() for interaction_values in interactions.values()]
    )
    max_value = max(
        [np.abs(interaction_values.values).max() for interaction_values in interactions.values()]
    )
    min_max_interactions = None
    if ADJUST_MIN_MAX:
        min_max_interactions = (min_value, max_value)

    # get the graph labels -------------------------------------------------------------------------
    graph = to_networkx(graph_instance, to_undirected=True)
    graph_labels, atom_names = None, None
    if DATASET_NAME == "MUTAG":
        atom_names = ["C", "N", "O", "F", "I", "Cl", "Br"]
    if DATASET_NAME == "Mutagenicity":
        atom_names = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
    if DATASET_NAME in ("Benzene", "FluorideCarbonyl", "AlkaneCarbonyl"):
        # link to source: https://github.com/google-research/graph-attribution/blob/main/graph_attribution/featurization.py#L37C1-L41C71
        atom_names = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "Na", "Ca", "I", "B", "H", "*"]

    if atom_names is not None:
        graph_labels = {
            node_id: atom_names[np.argmax(atom)]
            for node_id, atom in enumerate(graph_instance.x.numpy())
        }

    # plot full graph explanation ------------------------------------------------------------------
    moebius_values = interactions["n-SII"]
    print(moebius_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=moebius_values,
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
    title = "n-SII/MI Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_nSII.pdf"))
    plt.show()

    # plot 6-SII explanation -----------------------------------------------------------------------
    six_sii_values = interactions["6-SII"]
    print(six_sii_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=six_sii_values,
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
    title = "6-SII Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_6SII.pdf"))
    plt.show()

    # plot 3-SII explanation -----------------------------------------------------------------------
    three_sii_values = interactions["3-SII"]
    print(three_sii_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=three_sii_values,
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
    title = "3-SII Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_3SII.pdf"))
    plt.show()

    # plot 2-SII explanation -----------------------------------------------------------------------
    two_sii_values = interactions["2-SII"]
    print(two_sii_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=two_sii_values,
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
    title = "2-SII Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_2SII.pdf"))
    plt.show()

    # plot sv explanation --------------------------------------------------------------------------
    sv_values = interactions["SV"]
    print(sv_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=sv_values,
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
    title = "SV Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_SV.pdf"))
    plt.show()

    # plot the og graph ----------------------------------------------------------------------------
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=moebius_values,
        plot_explanation=False,
        n_interactions=N_INTERACTIONS,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        adjust_node_pos=ADJUST_NODE_POS,
        node_size_scaling=NODE_SIZE_SCALING,
        spring_k=SPRING_K,
        interaction_direction=INTERACTION_DIRECTION,
        draw_threshold=DRAW_THRESHOLD,
    )
    title = "Original Graph\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_graph.pdf"))
    plt.show()

    # plot 2-STII explanation ----------------------------------------------------------------------
    two_stii_values = interactions["2-STII"]
    print(two_stii_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=two_stii_values,
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
    title = "2-STII Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_2STII.pdf"))
    plt.show()

    # plot 6-STII explanation ----------------------------------------------------------------------
    six_stii_values = interactions["6-STII"]
    print(six_stii_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=six_stii_values,
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
    title = "6-STII Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_6STII.pdf"))
    plt.show()

    # plot SII-2 explanation ----------------------------------------------------------------------
    sii_2_values = interactions["SII-2"]
    print(sii_2_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=sii_2_values,
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
    title = "SII-2 Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_SII2.pdf"))
    plt.show()

    # plot SII-6 explanation -----------------------------------------------------------------------
    sii_6_values = interactions["SII-6"]
    print(sii_6_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=sii_6_values,
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
    title = "SII-6 Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_SII6.pdf"))
    plt.show()

    # plot 2-FSII explanation ----------------------------------------------------------------------
    two_fsii_values = interactions["2-FSII"]
    print(two_fsii_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=two_fsii_values,
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
    title = "2-FSII Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_2FSII.pdf"))
    plt.show()

    # plot 6-FSII explanation ----------------------------------------------------------------------
    six_fsii_values = interactions["6-FSII"]
    print(six_fsii_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=six_fsii_values,
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
    title = "6-FSII Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
    if SAVE_FIG:
        plt.savefig(os.path.join(PLOT_DIR, f"{file_identifier}_plot_6FSII.pdf"))
    plt.show()
