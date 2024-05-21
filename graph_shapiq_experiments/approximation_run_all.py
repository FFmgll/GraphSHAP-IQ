"""This script runs baseline approximation methods on all datasets that we currently have."""

from approximation_run_baselines import approximate_baselines

if __name__ == "__main__":

    ITERATIONS = [
        # 1,
        2,
    ]
    MODEL_IDS = [
        # "GCN",
        # "GAT",
        "GIN",
    ]
    N_LAYERS = [
        2,
        # 3,
    ]
    INDICES = [
        "k-SII",
        # "SV",
    ]
    DATASETS = [
        "Mutagenicity",
        # "PROTEINS",
        # "BZR",
    ]

    for dataset_name in DATASETS:
        for index in INDICES:
            if index == "SV":
                MAX_ORDER = 1
                APPROXIMATORS_TO_RUN = [
                    "KernelSHAP",
                    "PermutationSamplingSV",
                    "SVARM",
                    "kADDSHAP",
                    "UnbiasedKernelSHAP",
                ]
            else:
                MAX_ORDER = 3
                APPROXIMATORS_TO_RUN = [
                    "PermutationSamplingSII",
                    "SHAPIQ",
                    "KernelSHAPIQ",
                    "InconsistentKernelSHAPIQ",
                    "SVARMIQ",
                ]
            for model_id in MODEL_IDS:
                for n_layer in N_LAYERS:
                    for approx_name in APPROXIMATORS_TO_RUN:
                        approx_run = [approx_name]
                        approximate_baselines(
                            dataset_name=dataset_name,
                            model_id=model_id,
                            n_layers=n_layer,
                            iterations=ITERATIONS,
                            index=index,
                            max_order=MAX_ORDER,
                            small_graph=False,
                            max_approx_budget=2**15,
                            approximators_to_run=approx_run,
                        )
