"""This module contains the TreeSHAPIQ explanation benchmark games."""

from typing import Optional, Union

import numpy as np

from ..setup import BenchmarkSetup, get_x_explain
from .base import TreeSHAPIQXAI


class CaliforniaHousing(TreeSHAPIQXAI):

    def __init__(
        self,
        x: Optional[Union[np.ndarray, int]] = None,
        model_name: str = "decision_tree",
        index: str = "k-SII",
        class_label: Optional[int] = None,
        max_order: int = 2,
        min_order: int = 1,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:

        setup = BenchmarkSetup(
            dataset_name="california_housing",
            model_name=model_name,
            verbose=verbose,
        )

        # get x_explain
        x = get_x_explain(x, setup.x_test)

        # call the super constructor
        super().__init__(
            x=x,
            tree_model=setup.model,
            normalize=normalize,
            index=index,
            class_label=class_label,
            max_order=max_order,
            min_order=min_order,
        )
