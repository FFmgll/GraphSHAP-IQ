"""This module contains the TreeExplainer class making use of the TreeSHAPIQ algorithm for
computing any-order Shapley Interactions for tree ensembles."""

import copy
from typing import Any, Optional, Union

import numpy as np

from shapiq.explainer._base import Explainer
from shapiq.interaction_values import InteractionValues

from .graphshapiq import GraphSHAPIQ



class GraphNodeExplainer(Explainer):
    def __init__(
        self,
        model,
        max_order: int = 2,
        min_order: int = 0,
        interaction_type: str = "k-SII",
        class_label: Optional[int] = None,
        output_type: str = "raw",
        **kwargs
    ) -> None:

        super().__init__(model)

        self._max_order: int = max_order
        self._min_order: int = min_order
        self._class_label: Optional[int] = class_label
        self._output_type: str = output_type


    def explain(self, x: np.ndarray) -> InteractionValues:
        # run graphshapiq
        interaction_values: list[InteractionValues] = []

        return interaction_values
