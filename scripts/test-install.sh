#!/bin/bash
# Test installation script for docling-onnx-models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🧪 Testing docling-onnx-models installation...${NC}"

# Create temporary virtual environment
TEMP_VENV=$(mktemp -d)/test-env
echo -e "${BLUE}📁 Creating test environment: $TEMP_VENV${NC}"
python -m venv "$TEMP_VENV"

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source "$TEMP_VENV/Scripts/activate"
else
    source "$TEMP_VENV/bin/activate"
fi

# Install package
echo -e "${BLUE}📦 Installing package...${NC}"
if [ -f "dist/*.whl" ]; then
    pip install dist/*.whl
else
    echo -e "${RED}❌ No wheel file found in dist/. Run build script first.${NC}"
    exit 1
fi

# Test basic import
echo -e "${BLUE}🔍 Testing basic import...${NC}"
python -c "
import docling_onnx_models
print(f'✅ Successfully imported docling_onnx_models version: {docling_onnx_models.__version__}')
"

# Test provider detection
echo -e "${BLUE}🔍 Testing provider detection...${NC}"
python -c "
from docling_onnx_models.common import get_optimal_providers
import platform

print(f'Platform: {platform.system()}')
providers = get_optimal_providers('auto')
print(f'Auto-selected providers: {providers}')

if not providers:
    raise RuntimeError('No providers selected')
    
if 'CPUExecutionProvider' not in providers:
    raise RuntimeError('CPU provider missing')
    
print('✅ Provider detection working correctly')
"

# Test layout predictor
echo -e "${BLUE}🔍 Testing layout predictor initialization...${NC}"
python -c "
from docling_onnx_models.layoutmodel import LayoutPredictor
from docling_onnx_models.layoutmodel.layout_config import LayoutConfig

config = LayoutConfig()
predictor = LayoutPredictor(config)
print('✅ Layout predictor initialized successfully')
"

# Test table predictor  
echo -e "${BLUE}🔍 Testing table predictor initialization...${NC}"
python -c "
from docling_onnx_models.tableformer import TableFormerPredictor
from docling_onnx_models.tableformer.table_config import TableConfig

config = TableConfig()
predictor = TableFormerPredictor(config)
print('✅ Table predictor initialized successfully')
"

# Test figure classifier
echo -e "${BLUE}🔍 Testing figure classifier initialization...${NC}"
python -c "
from docling_onnx_models.document_figure_classifier import DocumentFigureClassifier
from docling_onnx_models.document_figure_classifier.figure_config import FigureConfig

config = FigureConfig()
classifier = DocumentFigureClassifier(config)
print('✅ Figure classifier initialized successfully')
"

# Cleanup
deactivate
rm -rf "$TEMP_VENV"

echo ""
echo -e "${GREEN}🎉 All tests passed successfully!${NC}"
echo -e "${GREEN}✅ Package is ready for distribution${NC}"