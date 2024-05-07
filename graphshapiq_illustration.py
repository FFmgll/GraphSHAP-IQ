from shapiq.explainer.graph.plotting import GraphPlotter
from shapiq.explainer.graph import get_explanation_instances
import numpy as np

if __name__ == "__main__":
    plotter = GraphPlotter()
    explanation_instances = get_explanation_instances("MUTAG")

    save_path = "results/illustration"

    counter = 0
    for data_id, instance in enumerate(explanation_instances):
        if instance.num_nodes >= 20:
            if counter >= 5:
                continue
            masked_nodes = np.random.choice(
                range(instance.num_nodes), size=int(instance.num_nodes / 3), replace=False
            )
            plotter.plot_graph(instance, masked_nodes, save_path + "/" + str(data_id) + ".png")
            counter += 1
