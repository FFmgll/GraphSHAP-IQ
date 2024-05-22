# Exact Computation of Any-Order Shapley Interactions on Graph Neural Networks
This repository contains the code supplement for the paper "Exact Computation of Any-Order Shapley Interactions on Graph Neural Networks"

## Requirements
The code is written and tested with Python 3.10 and the requirements are listed in `requirements_graphshapiq.txt`. You can install the requirements using the following command:
```bash
pip install -r requirements_graphshapiq.txt
```

## Usage
The implementation is organized in the shapiq package. For implementation details regarding the GraphSHAP-IQ implementation we refer to `shapiq/graph_shap_iq.py`. The `shapiq` package also contains the implementation of the GraphSHAP algorithm in `shapiq/explainer/graph`.
To validate the experiments we refer to `graph_shapiq_experiments`. 
To see all plots and data please refer to the `plots_in_paper` folder.

## Note
The .interaction_values files created for the approximation experiments exceed 250MB and, thus, are not included in the repository. If you want to reproduce the results, please run the experiments in `graph_shapiq_experiments` and the files will be created.
