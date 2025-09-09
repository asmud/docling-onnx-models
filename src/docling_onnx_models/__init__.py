#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""
ONNX-based models for Docling document processing.

This package provides ONNX Runtime implementations of the models
originally found in docling-ibm-models, offering improved performance
and cross-platform compatibility.
"""

__version__ = "1.0.0"
__all__ = [
    "layoutmodel",
    "tableformer",
    "document_figure_classifier_model",
    "code_formula_model",
    "common",
]
