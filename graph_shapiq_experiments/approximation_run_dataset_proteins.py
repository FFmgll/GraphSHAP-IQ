"""This script runs baseline approximation methods on the PROTEINS dataset."""

from shapiq.approximator import (
    KernelSHAPIQ,
    PermutationSamplingSII,
    PermutationSamplingSV,
    KernelSHAP,
)
from approximation_run_baselines import approximate_baselines


if __name__ == "__main__":

    INDEX = "SV"
    MAX_ORDER = 2
    ITERATIONS = 2
    MODEL_IDS = [
        "GCN",
        "GAT",
        "GIN",
    ]
    N_LAYERS = 2

    if INDEX == "SV":
        MAX_ORDER = 1
        APPROXIMATORS_TO_RUN = [
            PermutationSamplingSV.__name__,
            KernelSHAP.__name__,
        ]
    else:
        APPROXIMATORS_TO_RUN = [
            PermutationSamplingSII.__name__,
            KernelSHAPIQ.__name__,
        ]

    for model_id in MODEL_IDS:
        approximate_baselines(
            dataset_name="PROTEINS",
            model_id=model_id,
            n_layers=N_LAYERS,
            iterations=[1, 2],
            index=INDEX,
            max_order=MAX_ORDER,
            small_graph=False,
            max_approx_budget=2**15,
            approximators_to_run=APPROXIMATORS_TO_RUN,
        )
