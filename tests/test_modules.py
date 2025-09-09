"""Tests for module imports and basic functionality."""

import pytest


class TestModuleImports:
    """Test that all modules can be imported."""

    def test_import_code_formula_model(self):
        """Test code formula model import."""
        from docling_onnx_models.code_formula_model import __all__
        assert isinstance(__all__, list)

    def test_import_tableformer_utils(self):
        """Test tableformer utils import."""
        from docling_onnx_models.tableformer.utils import __all__
        assert isinstance(__all__, list)

    def test_import_tableformer_main(self):
        """Test tableformer main module import."""
        from docling_onnx_models import tableformer
        assert tableformer is not None

    def test_import_document_classifier_main(self):
        """Test document classifier main module import.""" 
        from docling_onnx_models import document_figure_classifier_model
        assert document_figure_classifier_model is not None

    def test_import_tableformer_data_management(self):
        """Test tableformer data management import."""
        from docling_onnx_models.tableformer import data_management
        assert data_management is not None


class TestPackageStructure:
    """Test package structure and exports."""

    def test_main_package_all(self):
        """Test main package __all__ exports."""
        from docling_onnx_models import __all__
        
        assert isinstance(__all__, list)
        assert "layoutmodel" in __all__
        assert "tableformer" in __all__
        assert "document_figure_classifier_model" in __all__
        assert "code_formula_model" in __all__
        assert "common" in __all__

    def test_version_available(self):
        """Test that version is available."""
        import docling_onnx_models
        
        assert hasattr(docling_onnx_models, "__version__")
        assert isinstance(docling_onnx_models.__version__, str)
        assert len(docling_onnx_models.__version__) > 0