#!/bin/bash
# Publish script for docling-onnx-models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Publishing docling-onnx-models package...${NC}"

# Check if we're on the main branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo -e "${RED}❌ Not on main branch. Current: $current_branch${NC}"
    echo "Switch to main branch before publishing"
    exit 1
fi

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}❌ Working directory not clean${NC}"
    echo "Commit or stash changes before publishing"
    exit 1
fi

# Check if version is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: $0 <version> [--test]${NC}"
    echo "Example: $0 1.0.0"
    echo "         $0 1.0.0 --test (publishes to test PyPI)"
    exit 1
fi

VERSION=$1
TEST_MODE=${2:-""}

# Validate version format
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}❌ Invalid version format. Use semantic versioning (e.g., 1.0.0)${NC}"
    exit 1
fi

echo -e "${GREEN}📋 Publishing version: $VERSION${NC}"

# Build the package
echo -e "${GREEN}🔨 Building package...${NC}"
./scripts/build.sh

# Create and push tag
echo -e "${GREEN}🏷️  Creating git tag...${NC}"
git tag "v$VERSION"
git push origin "v$VERSION"

# Publish to PyPI
if [ "$TEST_MODE" == "--test" ]; then
    echo -e "${YELLOW}🧪 Publishing to Test PyPI...${NC}"
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    echo -e "${GREEN}✅ Published to Test PyPI!${NC}"
    echo -e "${GREEN}Test installation: pip install --index-url https://test.pypi.org/simple/ docling-onnx-models==$VERSION${NC}"
else
    echo -e "${GREEN}📦 Publishing to PyPI...${NC}"
    twine upload dist/*
    echo -e "${GREEN}✅ Published to PyPI!${NC}"
    echo -e "${GREEN}Installation: pip install docling-onnx-models==$VERSION${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Publication completed successfully!${NC}"
echo -e "${GREEN}📊 Package info: https://pypi.org/project/docling-onnx-models/$VERSION${NC}"