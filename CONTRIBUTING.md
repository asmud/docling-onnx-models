# Contributing to Docling ONNX Models

Thank you for your interest in contributing to Docling ONNX Models! This document provides guidelines and information for contributors.

## Getting Started

### Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/docling-project/docling-onnx-models.git
cd docling-onnx-models
```

2. **Set up development environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

3. **Install pre-commit hooks:**
```bash
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=docling_onnx_models

# Run specific test file
pytest tests/test_layout_predictor.py

# Run tests in parallel
pytest -n auto
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Sort imports  
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Maximum line length: 88 characters

### Type Hints

- Use type hints for all public functions and methods
- Use `typing` module for complex types
- Add `py.typed` marker for type information

### Documentation

- Use Google-style docstrings
- Document all public APIs
- Include usage examples in docstrings
- Update README.md for user-facing changes

### Testing

- Write unit tests for new functionality
- Maintain test coverage above 80%
- Use descriptive test names
- Mock external dependencies appropriately

## Contribution Process

### 1. Issue Discussion

- Check existing issues before creating new ones
- Discuss major changes in GitHub issues first
- Use issue templates when available

### 2. Development

- Create a feature branch from `main`
- Follow the coding standards
- Write or update tests
- Update documentation

### 3. Pull Request

- Create a pull request against `main`
- Use the PR template
- Ensure all CI checks pass
- Request review from maintainers

### 4. Review Process

- Address reviewer feedback
- Maintain clean commit history
- Squash commits if requested

## Types of Contributions

### Bug Reports

Use the bug report template and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- System information
- Relevant code/logs

### Feature Requests

Use the feature request template and include:
- Clear description of the feature
- Use cases and motivation
- Proposed implementation approach
- Breaking change considerations

### Code Contributions

Types of contributions welcome:
- Bug fixes
- New ONNX model implementations
- Performance optimizations
- Documentation improvements
- Test coverage improvements
- CI/CD enhancements

### Performance Improvements

When contributing performance optimizations:
- Include benchmarks showing improvement
- Document any trade-offs
- Ensure backward compatibility
- Add appropriate tests

## Model Integration

### Adding New ONNX Models

1. **Create predictor class:**
   - Inherit from `BaseONNXPredictor`
   - Implement required abstract methods
   - Add proper error handling

2. **Add model configuration:**
   - Create configuration class
   - Add to model specs
   - Document parameters

3. **Write comprehensive tests:**
   - Unit tests for predictor
   - Integration tests
   - Performance benchmarks

4. **Update documentation:**
   - API documentation
   - Usage examples
   - README updates

### Model Conversion Guidelines

- Document conversion process
- Provide conversion scripts
- Validate ONNX model outputs
- Test across different platforms

## Release Process

### Version Management

- Follow [Semantic Versioning](https://semver.org/)
- Update `CHANGELOG.md`
- Tag releases appropriately

### Publishing

1. **Prepare release:**
   - Update version in `pyproject.toml`
   - Update `CHANGELOG.md`
   - Create release notes

2. **Build and test:**
   ```bash
   python -m build
   twine check dist/*
   ```

3. **Publish:**
   ```bash
   twine upload dist/*
   ```

## Community Guidelines

### Code of Conduct

Please follow the [IBM Open Source Code of Conduct](https://github.com/IBM/code-of-conduct).

### Communication

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Share knowledge and experience

## Getting Help

- **Documentation:** Check README and docstrings
- **Issues:** Search existing GitHub issues
- **Discussions:** Use GitHub Discussions for questions
- **Email:** Contact maintainers for sensitive issues

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub repository insights

Thank you for contributing to Docling ONNX Models!