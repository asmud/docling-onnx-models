"""Basic tests to ensure package can be imported."""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import docling_onnx_models
    
    assert hasattr(docling_onnx_models, "__version__")


def test_common_import():
    """Test that common utilities can be imported."""
    from docling_onnx_models.common import get_optimal_providers
    
    # Test that we can call the function
    providers = get_optimal_providers("auto")
    assert isinstance(providers, list)
    assert len(providers) > 0
    assert "CPUExecutionProvider" in providers


def test_layout_model_import():
    """Test that layout model components can be imported."""
    from docling_onnx_models.layoutmodel import LayoutPredictor, LayoutLabels
    
    assert LayoutPredictor is not None
    assert LayoutLabels is not None