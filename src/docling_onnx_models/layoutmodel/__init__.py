#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""ONNX-based layout model implementations."""

from .labels import LayoutLabels
from .layout_predictor import LayoutPredictor

__all__ = ["LayoutPredictor", "LayoutLabels"]
