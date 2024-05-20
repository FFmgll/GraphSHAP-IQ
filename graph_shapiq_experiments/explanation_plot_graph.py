"""This script is used to visualize the explanation graphs for the GraphSHAP-IQ approximation."""
import copy
import os

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

    MODEL_TYPE = "GIN"
    DATASET_NAME = "Benzene"
    N_LAYER = 2
    DATA_ID = 2

    # for saving the plots
    file_identifier = "_".join([MODEL_TYPE, DATASET_NAME, str(N_LAYER), str(DATA_ID)])

    # plot parameter
    RANDOM_SEED = 4  # random seed for the graph layout
    COMPACTNESS = 10  # compactness of the graph layout
    SIZE_FACTOR = 2  # factor to scale the node sizes
    N_INTERACTIONS = 200  # number of interactions/explanations to plot for the graph
    CUBIC_SCALING = False  # uses cubic scaling for the node sizes (True) or not (False)
    ADJUST_MIN_MAX = True  # scales the explanation sizes across plots (True) or not (False)
    ADJUST_NODE_POS = False  # adjusts the node positions in the plots (True) or not (False)
    NODE_SIZE_SCALING = 1  # scales the node sizes in the plots
    SPRING_K = None  # (None) spring constant for the layout increase for more space between nodes

    RUN_MODEL = False
    SAVE_FIG = False
    PLOT_TITLE = False

    # increase font size for the plots and set bold
    if not PLOT_TITLE:
        plt.rcParams["font.size"] = 15
        # plt.rcParams["font.weight"] = "bold"

    # evaluate the model on the instance -----------------------------------------------------------

    explanation_instances = get_explanation_instances(DATASET_NAME)
    graph_instance = explanation_instances[DATA_ID]
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
    print("Model prediction: ", prediction)
    game = GraphGame(
        model,
        x_graph=graph_instance,
        class_id=predicted_class,
        max_neighborhood_size=model.n_layers,
        masking_mode="feature-removal",
        normalize=True,
        baseline=_compute_baseline_value(graph_instance),
        instance_id=int(DATA_ID),
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
        "3-SII": converter(index=INDEX, order=3),
        "2-SII": converter(index=INDEX, order=2),
        "SV": converter(index=INDEX, order=1),
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
    graph_labels = None
    if DATASET_NAME == "MUTAG":
        # make one-hot encoding of the atom types into labels
        mutag_atom_names = ["C", "N", "O", "F", "I", "Cl", "Br"]
        graph_labels = {
            node_id: mutag_atom_names[np.argmax(atom)]
            for node_id, atom in enumerate(graph_instance.x.numpy())
        }
    if DATASET_NAME == "Mutagenicity":
        # make one-hot encoding of the atom types into labels
        mutagenicity_atom_names = [
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
        graph_labels = {
            node_id: mutagenicity_atom_names[np.argmax(atom)]
            for node_id, atom in enumerate(graph_instance.x.numpy())
        }

    # plot full graph explanation ------------------------------------------------------------------
    moebius_values = interactions["n-SII"]
    print(moebius_values)
    fig, _ = explanation_graph_plot(
        graph=graph,
        interaction_values=moebius_values,
        plot_explanation=True,
        n_interactions=5,
        size_factor=SIZE_FACTOR,
        compactness=COMPACTNESS,
        random_seed=RANDOM_SEED,
        label_mapping=graph_labels,
        cubic_scaling=CUBIC_SCALING,
        min_max_interactions=min_max_interactions,
        adjust_node_pos=ADJUST_NODE_POS,
        node_size_scaling=NODE_SIZE_SCALING,
        spring_k=SPRING_K,
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
    )
    title = "2-SII Explanation\n" + title_suffix
    if PLOT_TITLE:
        plt.title(title)
        plt.tight_layout()
    else:
        plt.subplots_adjust(left=-0.02, right=1.02, bottom=-0.02, top=1.02)
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
