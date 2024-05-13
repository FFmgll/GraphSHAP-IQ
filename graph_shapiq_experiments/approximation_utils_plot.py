import copy
import os
from typing import Optional

import pandas as pd
from tqdm.asyncio import tqdm

from shapiq.games.benchmark import get_all_metrics
from shapiq.moebius_converter import MoebiusConverter
from shapiq import InteractionValues
from approximation_utils import create_results_overview_table


def load_interactions_to_plot(
    overview_table: pd.DataFrame,
    index: str,
    max_order: int,
) -> pd.DataFrame:
    """Load the interaction values of the approximations and the exact values from the GraphSHAP-IQ
    approximations and return the results as a list of dictionaries.

    Args:
        overview_table: The overview table to load the interaction values from.
        index: The index to use for the Möbius transformation.
        max_order: The maximum order to use for the Möbius transformation.

    Returns:
        The results to plot as a list of dictionaries.
    """

    results_to_plot: list[dict] = []
    # get exact values and transform them with Möbius
    exact_values_index: dict[str, InteractionValues] = {}  # instance_id -> InteractionValues
    instances_exact = overview_table[overview_table["exact"] == True][["instance_id", "file_path"]]
    for instance_id, file_path in tqdm(
        instances_exact.itertuples(index=False, name=None),
        desc="Transforming exact values",
        total=len(instances_exact),
    ):
        exact = InteractionValues.load(file_path)
        converter = MoebiusConverter(moebius_coefficients=exact)
        exact_values_index[instance_id] = converter(index=index, order=max_order)
    print(f"Found {len(exact_values_index)} exact values.")
    for instance_id, values in exact_values_index.items():
        n_players = values.n_players
        print(f"Instance {instance_id} with {n_players} players.")

    # get and transform the graph_shapiq values
    graph_shapiq_values_index: dict[str, InteractionValues] = {}  # instance_id -> InteractionValues
    gshap = overview_table[overview_table["approximation"] == "GraphSHAPIQ"][
        ["run_id", "instance_id", "budget", "file_path"]
    ]
    for run_id, instance_id, budget, file_path in tqdm(
        gshap.itertuples(index=False, name=None),
        desc="Transforming GraphSHAP-IQ values",
        total=len(gshap),
    ):
        if instance_id not in instances_exact["instance_id"].values:
            print(f"Skipping {instance_id} because exact values are not computed.")
            continue
        graph_shapiq = InteractionValues.load(file_path)
        converter = MoebiusConverter(moebius_coefficients=graph_shapiq)
        graph_shapiq_values_index[instance_id] = converter(index=index, order=max_order)

        metrics = get_all_metrics(
            ground_truth=exact_values_index[instance_id],
            estimated=graph_shapiq_values_index[instance_id],
        )
        results_to_plot.append(
            {
                "run_id": run_id,
                "instance_id": instance_id,
                "approximation": "GraphSHAPIQ",
                "budget": budget,
                "index": index,
                "max_order": max_order,
                "min_order": 0,
                **metrics,
            }
        )

    # add approximator
    for approx_name in ["PermutationSamplingSII", "KernelSHAPIQ", "SVARMIQ"]:
        approx: dict[str, InteractionValues] = {}  # instance_id - InteractionValues
        kshap = overview_table[overview_table["approximation"] == approx_name][
            ["run_id", "instance_id", "budget", "file_path"]
        ]
        for run_id, instance_id, budget, file_path in tqdm(
            kshap.itertuples(index=False, name=None),
            desc=f"Loading {approx_name} values",
            total=len(kshap),
        ):
            if instance_id not in instances_exact["instance_id"].values:
                print(f"Skipping {instance_id} because exact values are not computed.")
                continue
            approx[instance_id] = InteractionValues.load(file_path)
            metrics = get_all_metrics(
                ground_truth=exact_values_index[instance_id],
                estimated=approx[instance_id],
            )
            results_to_plot.append(
                {
                    "run_id": run_id,
                    "instance_id": instance_id,
                    "approximation": approx_name,
                    "budget": approx[instance_id].estimation_budget,
                    "index": index,
                    "max_order": max_order,
                    "min_order": 0,
                    **metrics,
                }
            )
    return pd.DataFrame(results_to_plot)


def get_plot_df(
    index: str,
    max_order: int,
    dataset_name: str,
    n_layers: int,
    model_id: str,
    small_graph: bool,
    load_from_csv: bool = False,
    max_interaction_sizes_to_drop: Optional[int] = None,
) -> pd.DataFrame:
    """Get the DataFrame for the plot."""

    file_name = "plot_csv"
    file_name += f"_{dataset_name}_{model_id}_{n_layers}_{small_graph}_{index}_{max_order}.csv"

    # get the overview table for the specified parameters
    overview_table = create_results_overview_table()
    overview_table = copy.deepcopy(
        overview_table[
            (overview_table["dataset_name"] == dataset_name)
            & (overview_table["n_layers"] == n_layers)
            & (overview_table["model_id"] == model_id)
            & (overview_table["small_graph"] == small_graph)
        ]
    )

    if not load_from_csv or not os.path.exists(file_name):
        plot_df = load_interactions_to_plot(overview_table, index, max_order)
        plot_df.to_csv(file_name, index=False)
    else:
        plot_df = pd.read_csv(file_name)

    # inner join with the overview table and drop all duplicate columns from the overview table
    merge_columns = ["run_id"]
    plot_df = pd.merge(plot_df, overview_table, on=merge_columns, how="inner")
    for column in plot_df.columns:
        if column.endswith("_y"):
            plot_df = plot_df.drop(column, axis=1)
        if column.endswith("_x"):
            plot_df = plot_df.rename(columns={column: column[:-2]})

    # for each instance_id, the top max_interaction_sizes_to_drop
    if max_interaction_sizes_to_drop is not None:
        for instance_id in plot_df["instance_id"].unique():
            instance_df = plot_df[plot_df["instance_id"] == instance_id]
            max_size = instance_df["max_interaction_size"].max()
            for size in range(max_size, max_size - max_interaction_sizes_to_drop, -1):
                plot_df = plot_df.drop(
                    plot_df[
                        (plot_df["instance_id"] == instance_id)
                        & (plot_df["max_interaction_size"] == size)
                    ].index
                )

    return plot_df
