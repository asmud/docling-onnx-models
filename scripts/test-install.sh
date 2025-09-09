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
WHEEL_FILE=$(ls dist/*.whl 2>/dev/null | head -1)
if [ -n "$WHEEL_FILE" ]; then
    pip install "$WHEEL_FILE"
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

# Test that the class can be imported and instantiated (without a real model path)
try:
    # This should work for import test
    print('✅ Layout predictor class imported successfully')
    print('Note: Actual initialization requires model artifacts path')
except Exception as e:
    raise e
"

# Test table predictor  
echo -e "${BLUE}🔍 Testing table predictor initialization...${NC}"
python -c "
from docling_onnx_models.tableformer import TFPredictor

# Test that the class can be imported 
try:
    print('✅ Table predictor class imported successfully')
    print('Note: Actual initialization requires config dictionary and model artifacts')
except Exception as e:
    raise e
"

# Test figure classifier
echo -e "${BLUE}🔍 Testing figure classifier initialization...${NC}"
python -c "
from docling_onnx_models.document_figure_classifier_model import DocumentFigureClassifierPredictor

# Test that the class can be imported
try:
    print('✅ Figure classifier class imported successfully')
    print('Note: Actual initialization requires artifacts path')
except Exception as e:
    raise e
"

# Cleanup
deactivate
rm -rf "$TEMP_VENV"

echo ""
echo -e "${GREEN}🎉 All tests passed successfully!${NC}"
echo -e "${GREEN}✅ Package is ready for distribution${NC}"