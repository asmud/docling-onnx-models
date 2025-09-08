# Docling ONNX Models

ONNX Runtime implementations for Docling AI models, providing improved performance and cross-platform compatibility.

## Overview

`docling-onnx-models` is a drop-in replacement for the `docling-ibm-models` package, offering the same APIs but powered by ONNX Runtime. This provides several advantages:

- **Improved Performance**: ONNX Runtime optimizations for better inference speed
- **Cross-Platform**: Consistent behavior across different operating systems
- **Hardware Acceleration**: Support for CPU, CUDA, and other execution providers
- **Reduced Dependencies**: Lighter weight than full PyTorch models

## Installation

```bash
pip install docling-onnx-models
```

For GPU support:
```bash
pip install docling-onnx-models[gpu]
```

## Supported Models

### Layout Model
- **Input**: Document page images
- **Output**: Layout element detection (text, tables, figures, etc.)
- **Compatible with**: `docling.models.layout_model.LayoutModel`

### Document Figure Classifier
- **Input**: Figure/image regions
- **Output**: Classification into 16 figure types (charts, logos, maps, etc.)
- **Compatible with**: `docling.models.document_picture_classifier.DocumentPictureClassifier`

### Table Structure Predictor
- **Input**: Table region images and token data
- **Output**: Table structure with rows, columns, and cell relationships
- **Compatible with**: `docling.models.table_structure_model.TableStructureModel`

## Usage

### Basic Usage

The ONNX models are designed as drop-in replacements for the original models:

```python
# Instead of:
# from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

# Use:
from docling_onnx_models.layoutmodel.layout_predictor import LayoutPredictor

# Same API
predictor = LayoutPredictor(
    artifact_path="/path/to/onnx/model/directory",
    device="cpu",
    num_threads=4
)

predictions = predictor.predict_batch(images)
```

### With Docling

To use ONNX models with Docling, ensure your model directories contain ONNX files:

```
model_directory/
├── model.onnx              # ONNX model file
├── config.json             # Model configuration
├── preprocessor_config.json # Preprocessing config
└── ...                     # Other model files
```

The models will be automatically detected and used by Docling when available.

### Custom Execution Providers

```python
from docling_onnx_models.layoutmodel.layout_predictor import LayoutPredictor

# Use specific execution providers
predictor = LayoutPredictor(
    artifact_path="/path/to/model",
    device="cuda",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

## Model Conversion

If you have PyTorch models that need to be converted to ONNX:

```python
# Example for converting a layout model
import torch
from transformers import AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained("path/to/pytorch/model")
model.eval()

# Dummy input (adjust dimensions as needed)
dummy_input = torch.randn(1, 3, 640, 640)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)
```

## Performance Tips

1. **Use appropriate execution providers** based on your hardware
2. **Set optimal thread counts** for CPU inference
3. **Batch processing** when possible for better throughput
4. **Model quantization** for reduced memory usage (if supported)

## API Compatibility

This package maintains full API compatibility with `docling-ibm-models`:

- All method signatures are identical
- Input/output formats are preserved  
- Configuration files use the same structure
- Error handling behavior is consistent

## Requirements

- Python 3.10+
- ONNX Runtime 1.15.0+
- NumPy 1.21.0+
- Pillow 8.3.0+
- OpenCV 4.5.0+

## Contributing

Please see the main [Docling repository](https://github.com/docling-project/docling) for contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.