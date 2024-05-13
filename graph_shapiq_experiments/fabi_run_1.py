"""This module runs baseline approximation methods on the same settings as the GraphSHAP-IQ
approximations."""

from shapiq.approximator import (
    KernelSHAPIQ,
    SHAPIQ,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    PermutationSamplingSII,
)
from approximation_run_baselines import approximate_baselines


if __name__ == "__main__":

    APPROXIMATORS_TO_RUN = [
        KernelSHAPIQ.__name__,
        PermutationSamplingSII.__name__,
        # SHAPIQ.__name__,
        # SVARMIQ.__name__,
        # InconsistentKernelSHAPIQ.__name__
    ]

    approximate_baselines(
        dataset_name="PROTEINS",
        model_id="GIN",
        n_layers=3,
        iterations=2,
        index="k-SII",
        max_order=2,
        small_graph=False,
        max_approx_budget=2**15,
        approximators_to_run=APPROXIMATORS_TO_RUN,
    )
