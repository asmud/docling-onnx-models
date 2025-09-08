# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of docling-onnx-models
- ONNX Runtime implementations for Docling AI models
- CoreML ExecutionProvider support for macOS optimization
- CUDA ExecutionProvider support for GPU acceleration
- Intelligent execution provider selection
- Layout model with ONNX backend
- Document figure classifier with ONNX backend
- Table structure predictor with ONNX backend
- Drop-in replacement APIs for docling-ibm-models
- Comprehensive model detection and validation utilities
- Auto-fallback to PyTorch when ONNX models unavailable
- Thread-safe model loading with proper error handling
- Batch processing optimizations
- Cross-platform compatibility (Windows, macOS, Linux)

### Changed
- N/A (Initial release)

### Deprecated
- N/A (Initial release)

### Removed
- N/A (Initial release)

### Fixed
- N/A (Initial release)

### Security
- N/A (Initial release)

## [1.0.0] - 2024-01-XX

### Added
- Initial stable release with full ONNX Runtime support

---

## Release Notes Template

When making a new release:

1. Update the version in `pyproject.toml`
2. Update this changelog
3. Create a git tag with the version number
4. Build and publish to PyPI

### Version Format
- Major.Minor.Patch (e.g., 1.0.0)
- Follow semantic versioning principles
- Pre-release versions: 1.0.0a1, 1.0.0b1, 1.0.0rc1