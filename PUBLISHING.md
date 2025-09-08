# Publishing Guide for Docling ONNX Models

This guide covers the complete process for packaging and publishing `docling-onnx-models` to PyPI.

## Quick Start

```bash
# 1. Build package
./scripts/build.sh

# 2. Test installation
./scripts/test-install.sh

# 3. Publish (replace 1.0.0 with actual version)
./scripts/publish.sh 1.0.0
```

## Prerequisites

### System Requirements

- Python 3.10+ with `pip` and `venv`
- Git with clean working directory
- Access to PyPI account with API token

### Required Tools

```bash
pip install build twine setuptools-scm
```

### PyPI Authentication

1. **Create PyPI account**: https://pypi.org/account/register/
2. **Generate API token**: https://pypi.org/manage/account/token/
3. **Configure credentials**:
   ```bash
   # Create ~/.pypirc
   [pypi]
   username = __token__
   password = pypi-your-api-token-here
   
   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-your-test-api-token-here
   ```

## Step-by-Step Publishing Process

### 1. Prepare Release

```bash
# Ensure clean working directory
git status

# Switch to main branch
git checkout main
git pull origin main

# Update version in pyproject.toml if needed
# (setuptools_scm handles versioning automatically from git tags)
```

### 2. Build Package

```bash
./scripts/build.sh
```

This script:
- Cleans previous builds
- Installs build dependencies
- Creates wheel and source distributions
- Validates package integrity
- Lists build artifacts

### 3. Test Installation

```bash
./scripts/test-install.sh
```

This script:
- Creates isolated test environment
- Installs the built package
- Tests all major components
- Verifies provider detection works
- Cleans up test environment

### 4. Publish Package

#### Test Publication (Recommended)

```bash
./scripts/publish.sh 1.0.0 --test
```

- Publishes to Test PyPI first
- Allows verification before main publication
- Test installation: `pip install --index-url https://test.pypi.org/simple/ docling-onnx-models`

#### Production Publication

```bash
./scripts/publish.sh 1.0.0
```

- Creates git tag `v1.0.0`
- Pushes tag to remote
- Publishes to PyPI
- Package available via: `pip install docling-onnx-models`

## Manual Process (Alternative)

If you prefer manual control:

### Build

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build package
python -m build

# Check package
twine check dist/*
```

### Test Locally

```bash
# Create test environment
python -m venv test-env
source test-env/bin/activate  # Windows: test-env\Scripts\activate

# Install and test
pip install dist/*.whl
python -c "import docling_onnx_models; print(docling_onnx_models.__version__)"

# Cleanup
deactivate
rm -rf test-env
```

### Publish

```bash
# Create git tag
git tag v1.0.0
git push origin v1.0.0

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Version Management

This package uses `setuptools_scm` for automatic versioning:

- **Development versions**: `1.0.0.dev123+gabcdef` (from commits)
- **Release versions**: `1.0.0` (from git tags)

### Creating Releases

```bash
# Create release tag
git tag v1.0.0

# Push tag
git push origin v1.0.0

# Build will automatically use tag version
python -m build
```

### Version Patterns

- **Major**: Breaking changes (`1.0.0` → `2.0.0`)
- **Minor**: New features, backward compatible (`1.0.0` → `1.1.0`)
- **Patch**: Bug fixes (`1.0.0` → `1.0.1`)
- **Pre-release**: `1.0.0a1`, `1.0.0b1`, `1.0.0rc1`

## GitHub Actions (Automated)

The repository includes GitHub Actions workflows:

### CI Workflow (`.github/workflows/ci.yml`)

Triggered on: Push to `main`/`develop`, Pull Requests

- Tests across Python 3.10, 3.11, 3.12
- Tests on Ubuntu, Windows, macOS
- Code quality checks (black, isort, flake8, mypy)
- ONNX provider testing
- Package building and validation

### Release Workflow (`.github/workflows/release.yml`)

Triggered on: Git tags (`v*`) or manual dispatch

- Builds package on Ubuntu
- Tests installation across platforms
- Publishes to Test PyPI (manual trigger)
- Publishes to PyPI (tag trigger)
- Creates GitHub release

#### Manual Release via GitHub

1. Go to: `Actions` → `Release`
2. Click: `Run workflow`
3. Select: Branch and options
4. Run workflow

#### Automated Release via Tag

```bash
git tag v1.0.0
git push origin v1.0.0
# GitHub Actions automatically builds and publishes
```

## Package Verification

After publishing, verify the package:

```bash
# Install from PyPI
pip install docling-onnx-models

# Test basic functionality
python -c "
import docling_onnx_models
print(f'Version: {docling_onnx_models.__version__}')

from docling_onnx_models.common import get_optimal_providers
print(f'Providers: {get_optimal_providers()}')
"
```

## Troubleshooting

### Common Issues

1. **Authentication Error**:
   - Verify PyPI API token in `~/.pypirc`
   - Check token permissions

2. **Package Exists Error**:
   - Cannot overwrite existing versions
   - Increment version number

3. **Build Failures**:
   - Check Python version compatibility
   - Verify all dependencies are available
   - Clean build directory: `rm -rf dist/ build/`

4. **Import Errors**:
   - Verify MANIFEST.in includes all necessary files
   - Check package structure in built wheel

### Debug Build

```bash
# Verbose build output
python -m build --verbose

# Inspect wheel contents
python -m zipfile -l dist/*.whl

# Test in isolated environment
python -m venv debug-env
source debug-env/bin/activate
pip install dist/*.whl
python -c "import docling_onnx_models; print(dir(docling_onnx_models))"
```

## Security Considerations

- **Never commit API tokens** to git
- **Use environment variables** for sensitive data in CI
- **Verify package contents** before publishing
- **Monitor security advisories** for dependencies

## Support

- **Issues**: https://github.com/docling-project/docling-onnx-models/issues
- **Discussions**: https://github.com/docling-project/docling-onnx-models/discussions
- **PyPI Project**: https://pypi.org/project/docling-onnx-models/