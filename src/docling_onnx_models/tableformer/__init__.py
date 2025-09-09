#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""ONNX-based tableformer implementation."""

from .data_management.tf_predictor import TFPredictor

__all__ = ["TFPredictor"]
