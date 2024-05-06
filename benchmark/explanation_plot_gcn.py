"""Tests the new explanation_plot function."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot.explanation_graph import explanation_graph
from shapiq.interaction_values import InteractionValues
from shapiq.moebius_converter import MoebiusConverter

if __name__ == "__main__":

    df = pd.read_csv("data/gtmoebius_GCN_MUTAG_3_True_True_4_11_11.csv")
    # rename first col from "" to "set"
    df = df.rename(columns={df.columns[0]: "set"})

    values = []
    lookup = {}
    n_players = 0
    for i, row in df.iterrows():
        val = float(row["gt"])
        coalition = row["set"]
        if coalition == "()":
            coalition = tuple()
        else:
            coalition = coalition.replace("(", "").replace(")", "")
            coalition_members = coalition.split(",")
            coalition_transformed = []
            for member in coalition_members:
                if member == "" or member == " " or member == ",":
                    continue
                try:
                    member = int(member)
                except ValueError:
                    member = member
                coalition_transformed.append(member)
            coalition = tuple(coalition_transformed)
        lookup[coalition] = len(values) - 1
        values.append(val)
        n_players = max(n_players, max(coalition, default=0))
    values = np.array(values)
    n_players += 1

    example_values = InteractionValues(
        n_players=n_players,
        values=values,
        index="Moebius",
        interaction_lookup=lookup,
        baseline_value=float(values[lookup[tuple()]]),
        min_order=0,
        max_order=n_players,
    )

    print("Loaded values.")
    converter = MoebiusConverter(N=set(range(n_players)), moebius_coefficients=example_values)
    example_values = converter(index="k-SII", order=n_players)
    print("Converted values.")
    print(example_values)

    print("Sum of values:", np.sum(example_values.values))

    fig, ax = explanation_graph(
        example_values,
        edges=[(0, 1), (2, 3), (0, 3), (0, 2), (1, 3), (4, 8), (7, 8), (7, 9), (3, 4), (5, 10)],
        random_seed=2,
        size_factor=1,
        plot_explanation=True,
        weight_factor=5,
        draw_threshold=0.03,
    )
    plt.show()
