"""Metrics for evaluating the performance of interaction values.
# TODO: make this more efficient
"""

from typing import Optional

import numpy as np
from scipy.stats import kendalltau

from ...interaction_values import InteractionValues
from ...utils.sets import powerset, count_interactions

__all__ = [
    "compute_kendall_tau",
    "compute_precision_at_k",
    "get_all_metrics",
]


def _remove_empty_value(interaction: InteractionValues) -> InteractionValues:
    """Manually sets the empty value to zero.

    Args:
        interaction: The interaction values to remove the empty value from.

    Returns:
        The interaction values without the empty value.
    """
    try:
        empty_index = interaction.interaction_lookup[()]
        interaction.values[empty_index] = 0
        return interaction
    except KeyError:
        return interaction


def compute_top_k_metrics(
    ground_truth: InteractionValues, estimated: InteractionValues, k: int = 10
) -> dict[str, float]:
    """Computes the MSE, MAE, and SSE between the top k interaction values in the ground truth.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        k: The top k interactions to consider. Defaults to 10.

    Returns:
        The mean squared error, mean absolute error, and sum of squared errors between the top k
        interaction values in the ground truth and estimated interaction values.
    """
    mse, mae, sse, sae = 0.0, 0.0, 0.0, 0.0
    top_k_gt = ground_truth.get_top_k(k=k, as_interaction_values=True)
    for interaction in top_k_gt.interaction_lookup.keys():
        gt_value = top_k_gt[interaction]
        diff = gt_value - estimated[interaction]
        mse += diff**2
        mae += abs(diff)
        sse += diff**2
        sae += abs(diff)
    mse /= k
    mae /= k
    return {f"MSE@{k}": mse, f"MAE@{k}": mae, f"SSE@{k}": sse, f"SAE@{k}": sae}


def compute_non_trivial_metrics(
    ground_truth: InteractionValues, estimated: InteractionValues
) -> dict[str, float]:
    """Computes the MSE, MAE, and SSE between two interaction values, excluding the trivial
    interactions (where the ground truth value is zero).

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.

    Returns:
        The mean squared error between the ground truth and estimated interaction values, excluding
        the trivial interactions.
    """
    mse, mae, sse, sae = 0.0, 0.0, 0.0, 0.0
    n_seen = 0
    for interaction, position in ground_truth.interaction_lookup.items():
        gt_value = ground_truth.values[position]
        if gt_value != 0:
            n_seen += 1
            diff = gt_value - estimated[interaction]
            mse += diff**2
            mae += abs(diff)
            sse += diff**2
            sae += abs(diff)
    mse /= n_seen
    mae /= n_seen
    return {"NonTrivialMSE": mse, "NonTrivialMAE": mae, "NonTrivialSSE": sse, "NonTrivialSAE": sae}


def compute_metrics(ground_truth: InteractionValues, estimated: InteractionValues) -> dict:
    """Compute the MSE (mean squared error), MAE (mean absolute error), SSE (sum of squared errors),
     and the SAE (sum of absolute errors) between two interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.

    Returns:
        The metrics between the ground truth and estimated interaction values.
    """
    difference = ground_truth - estimated
    diff_values = _remove_empty_value(difference).values
    n_values = count_interactions(
        ground_truth.n_players, ground_truth.max_order, ground_truth.min_order
    )
    metrics = {
        "MSE": np.sum(diff_values**2) / n_values,
        "MAE": np.sum(np.abs(diff_values)) / n_values,
        "SSE": np.sum(diff_values**2),
        "SAE": np.sum(np.abs(diff_values)),
    }
    return metrics


def compute_kendall_tau(
    ground_truth: InteractionValues, estimated: InteractionValues, k: int = None
) -> float:
    """Compute the Kendall Tau between two interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        k: The top-k ground truth values to consider. Defaults to `None`, which considers all
            interactions.

    Returns:
        The Kendall Tau between the ground truth and estimated interaction values.
    """
    # get the interactions as a sorted array
    gt_values, estimated_values = [], []
    for interaction in powerset(
        range(ground_truth.n_players),
        min_size=ground_truth.min_order,
        max_size=ground_truth.max_order,
    ):
        gt_values.append(ground_truth[interaction])
        estimated_values.append(estimated[interaction])
    # array conversion
    gt_values, estimated_values = np.array(gt_values), np.array(estimated_values)
    # sort the values
    gt_indices, estimated_indices = np.argsort(gt_values), np.argsort(estimated_values)
    if k is not None:
        gt_indices, estimated_indices = gt_indices[:k], estimated_indices[:k]
    # compute the Kendall Tau
    tau, _ = kendalltau(gt_indices, estimated_indices)
    return tau


def compute_precision_at_k(
    ground_truth: InteractionValues, estimated: InteractionValues, k: int = 10
) -> float:
    """Compute the precision at k between two interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        k: The top-k ground truth values to consider. Defaults to 10.

    Returns:
        The precision at k between the ground truth and estimated interaction values.
    """
    ground_truth_values = _remove_empty_value(ground_truth)
    estimated_values = _remove_empty_value(estimated)
    top_k, _ = ground_truth_values.get_top_k(k=k, as_interaction_values=False)
    top_k_estimated, _ = estimated_values.get_top_k(k=k, as_interaction_values=False)
    precision_at_k = len(set(top_k.keys()).intersection(set(top_k_estimated.keys()))) / k
    return precision_at_k


def get_all_metrics(
    ground_truth: InteractionValues,
    estimated: InteractionValues,
    order_indicator: Optional[str] = None,
) -> dict:
    """Get all metrics for the interaction values.

    Args:
        ground_truth: The ground truth interaction values.
        estimated: The estimated interaction values.
        order_indicator: The order indicator for the metrics. Defaults to None.

    Returns:
        The metrics as a dictionary.
    """
    if order_indicator is None:
        order_indicator = ""
    else:
        order_indicator += "_"

    metrics = {
        order_indicator + "Precision@10": compute_precision_at_k(ground_truth, estimated, k=10),
        order_indicator + "Precision@5": compute_precision_at_k(ground_truth, estimated, k=5),
        order_indicator + "KendallTau": compute_kendall_tau(ground_truth, estimated),
        order_indicator + "KendallTau@10": compute_kendall_tau(ground_truth, estimated, k=10),
        order_indicator + "KendallTau@50": compute_kendall_tau(ground_truth, estimated, k=50),
    }
    # get normal metrics
    metrics_normal = compute_metrics(ground_truth, estimated)
    metrics.update(metrics_normal)
    # get non-trivial metrics
    metrics_non_trivial = compute_non_trivial_metrics(ground_truth, estimated)
    metrics.update(metrics_non_trivial)
    # compute top-k metrics
    metrics.update(compute_top_k_metrics(ground_truth, estimated, k=10))
    metrics.update(compute_top_k_metrics(ground_truth, estimated, k=5))

    return metrics
