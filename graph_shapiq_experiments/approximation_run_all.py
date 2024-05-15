"""This script runs baseline approximation methods on all datasets that we currently have."""

from shapiq.approximator import (
    KernelSHAPIQ,
    PermutationSamplingSII,
    PermutationSamplingSV,
    KernelSHAP,
)
from approximation_run_baselines import approximate_baselines


if __name__ == "__main__":

    ITERATIONS = 2
    MODEL_IDS = [
        "GCN",
        "GAT",
        "GIN",
    ]
    N_LAYERS = [
        1,
        2,
        3,
    ]
    INDICES = [
        "k-SII",
        "SV",
    ]
    DATASETS = [
        "BZR",
        "PROTEINS",
        "Mutagenicity",
    ]

    for dataset_name in DATASETS:
        for index in INDICES:
            if index == "SV":
                MAX_ORDER = 1
                APPROXIMATORS_TO_RUN = [
                    PermutationSamplingSV.__name__,
                    KernelSHAP.__name__,
                ]
            else:
                MAX_ORDER = 2
                APPROXIMATORS_TO_RUN = [
                    PermutationSamplingSII.__name__,
                    KernelSHAPIQ.__name__,
                ]
            for model_id in MODEL_IDS:
                approximate_baselines(
                    dataset_name=dataset_name,
                    model_id=model_id,
                    n_layers=N_LAYERS,
                    iterations=ITERATIONS,
                    index=index,
                    max_order=MAX_ORDER,
                    small_graph=False,
                    max_approx_budget=2**15,
                    approximators_to_run=APPROXIMATORS_TO_RUN,
                )
