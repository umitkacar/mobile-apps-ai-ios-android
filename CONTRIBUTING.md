# Contributing to Mobile AI Apps

First off, thank you for considering contributing to Mobile AI Apps! 🎉

This document provides guidelines and instructions for contributing to this project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)
- [Style Guide](#style-guide)

---

## 📜 Code of Conduct

### Our Standards

- **Be respectful** and inclusive
- **Be collaborative** and supportive
- **Be professional** in all interactions
- **Focus on** what is best for the community
- **Show empathy** towards other community members

### Our Responsibilities

Project maintainers are responsible for clarifying standards and taking appropriate action in response to unacceptable behavior.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ installed
- Git installed
- Basic knowledge of Python, pytest, and modern Python tooling

### Quick Start

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/mobile-apps-ai-ios-android.git
cd mobile-apps-ai-ios-android

# 3. Add upstream remote
git remote add upstream https://github.com/umitkacar/mobile-apps-ai-ios-android.git

# 4. Run setup script
./scripts/setup.sh

# Or manual setup:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

---

## 🛠️ Development Setup

### 1. Install Development Dependencies

```bash
# Full development environment
pip install -e ".[dev,test,docs]"

# Or specific groups
pip install -e ".[dev]"   # Linting, formatting, etc.
pip install -e ".[test]"  # Testing tools
pip install -e ".[ml]"    # ML/AI dependencies
```

### 2. Setup Pre-commit Hooks

```bash
pre-commit install
pre-commit install --hook-type commit-msg

# Test the hooks
pre-commit run --all-files
```

### 3. Verify Setup

```bash
# Run all checks
make check

# Or individually
make lint          # Ruff linting
make format-check  # Black formatting
make type-check    # MyPy type checking
make test          # Run tests
```

---

## 🔨 Making Changes

### 1. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Or fix branch
git checkout -b fix/issue-number-description
```

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests
- `chore/` - Maintenance tasks

### 2. Make Your Changes

#### Code Changes

```python
# ✅ Good: Type hints, docstrings, clean code
def detect_objects(
    image: np.ndarray,
    confidence: float = 0.5,
) -> list[Detection]:
    """Detect objects in an image.

    Args:
        image: Input image as numpy array.
        confidence: Minimum confidence threshold.

    Returns:
        List of detected objects.

    Raises:
        TypeError: If image is not a numpy array.
        ValueError: If confidence is out of range.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1")

    # Implementation
    return []
```

#### Test Changes

```python
# ✅ Good: Clear test names, proper fixtures, assertions
def test_detect_objects_with_invalid_image(
    detector: YOLODetector,
) -> None:
    """Test detection with invalid image type."""
    with pytest.raises(TypeError, match="numpy array"):
        detector.detect("not_an_array")
```

### 3. Keep Changes Focused

- One feature/fix per pull request
- Keep pull requests small and focused
- Break large changes into smaller PRs
- Update tests and docs together with code

---

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_core.py

# Specific test function
pytest tests/test_core.py::test_config_creation

# With coverage
pytest --cov=src/mobile_ai --cov-report=html

# Parallel execution
pytest -n auto

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run only unit tests
pytest -m unit

# Run only fast tests
pytest -m "not slow"
```

### Writing Tests

#### Test Structure

```python
class TestYOURFeature:
    """Tests for YOUR feature."""

    def test_basic_functionality(self) -> None:
        """Test basic functionality works."""
        result = your_function(valid_input)
        assert result == expected_output

    def test_error_handling(self) -> None:
        """Test error handling."""
        with pytest.raises(ValueError, match="specific error"):
            your_function(invalid_input)

    @pytest.mark.slow
    def test_slow_operation(self) -> None:
        """Test slow operation (marked as slow)."""
        pass

    @pytest.mark.integration
    def test_integration(self) -> None:
        """Test integration with other components."""
        pass
```

#### Test Markers

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.ios` - iOS-specific
- `@pytest.mark.android` - Android-specific

#### Coverage Requirements

- **Core modules** (core.py, models.py): 90%+
- **Utility functions**: 80%+
- **CLI code**: Can be lower (functional tests)
- **Overall**: 60%+ required

---

## 🎨 Code Quality

### Automated Checks

```bash
# Run all checks
make check-all

# Individual checks
make lint          # Ruff linting
make format        # Black formatting (auto-fix)
make format-check  # Black check (no changes)
make type-check    # MyPy type checking
make audit         # Security audit
make test-cov      # Tests with coverage
```

### Pre-commit Hooks

Pre-commit runs automatically on `git commit`:

```bash
# Run manually on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files

# Skip hooks (emergency only!)
git commit --no-verify
```

### Code Style

#### Ruff

```bash
# Check
ruff check src tests

# Fix automatically
ruff check --fix src tests

# Show rule documentation
ruff rule E501
```

#### Black

```bash
# Check only
black --check src tests

# Format
black src tests
```

#### MyPy

```bash
# Type check
mypy src/mobile_ai

# Strict mode (default)
mypy --strict src/mobile_ai
```

---

## 📤 Submitting Changes

### 1. Commit Your Changes

#### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**

```bash
# Feature
git commit -m "feat(models): add YOLOv11 support

Implement YOLOv11 detector with optimized inference.

- Add YOLOv11 model class
- Update tests for new model
- Add documentation"

# Bug fix
git commit -m "fix(core): resolve Pydantic type issue

Fix Path import to work at runtime with Pydantic v2.

Fixes #123"

# Documentation
git commit -m "docs: update installation guide

Add troubleshooting section for common issues."
```

### 2. Push to Your Fork

```bash
# Push your branch
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### 4. Address Review Comments

```bash
# Make changes based on feedback
git add .
git commit -m "fix: address review comments"
git push origin feature/your-feature-name
```

### 5. After Merge

```bash
# Sync your fork
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# Delete feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

---

## 📁 Project Structure

```
mobile-apps-ai-ios-android/
├── src/mobile_ai/          # Main package
│   ├── __init__.py        # Package exports
│   ├── core.py            # Core functionality
│   ├── models.py          # AI models
│   └── cli.py             # CLI interface
├── tests/                  # Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── test_core.py       # Core tests
│   └── test_models.py     # Model tests
├── scripts/                # Development scripts
│   ├── setup.sh           # Setup script
│   ├── clean.sh           # Cleanup script
│   └── release.sh         # Release script
├── docs/                   # Documentation (if added)
├── .github/                # GitHub configs
├── pyproject.toml         # Project configuration
├── .pre-commit-config.yaml # Pre-commit hooks
├── Makefile               # Development commands
├── README.md              # Main documentation
├── CHANGELOG.md           # Version history
├── CONTRIBUTING.md        # This file
├── LESSONS-LEARNED.md     # Lessons learned
└── LICENSE                # License file
```

### Adding New Modules

```python
# src/mobile_ai/new_module.py
"""New module for X functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Your code here
```

```python
# src/mobile_ai/__init__.py
from mobile_ai.new_module import NewClass

__all__ = [
    ...,
    "NewClass",
]
```

```python
# tests/test_new_module.py
"""Tests for new module."""

from mobile_ai.new_module import NewClass

def test_new_functionality():
    """Test new functionality."""
    pass
```

---

## 📖 Style Guide

### Python Code

- **Line length**: 100 characters (Black default)
- **Imports**: Sorted with Ruff (isort-compatible)
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style, required for public APIs
- **Naming**:
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Constants: `UPPER_CASE`
  - Private: `_leading_underscore`

### Type Hints

```python
# ✅ Good
def process(
    data: np.ndarray,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    pass

# ❌ Bad
def process(data, threshold=0.5):
    pass
```

### Docstrings

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """One-line summary.

    Detailed description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ErrorType: When this error occurs.

    Example:
        >>> function_name(value1, value2)
        result
    """
```

### Error Handling

```python
# ✅ Good: Specific exceptions
if not isinstance(image, np.ndarray):
    raise TypeError("Image must be numpy array")

if len(image.shape) != 3:
    raise ValueError("Image must be 3D (H, W, C)")

# ❌ Bad: Generic exceptions
raise Exception("Something went wrong")
```

---

## 🎯 Tips for Good PRs

### DO ✅

- Write clear, descriptive commit messages
- Add tests for new features
- Update documentation
- Keep changes focused and small
- Respond to review comments promptly
- Be respectful and professional

### DON'T ❌

- Mix multiple unrelated changes
- Skip tests or documentation
- Ignore review feedback
- Force push after review started
- Commit sensitive information
- Break existing functionality

---

## 🆘 Getting Help

- **Questions**: Open a [Discussion](https://github.com/umitkacar/mobile-apps-ai-ios-android/discussions)
- **Bugs**: Open an [Issue](https://github.com/umitkacar/mobile-apps-ai-ios-android/issues)
- **Security**: Email maintainers privately
- **Documentation**: Check [README.md](README.md) and [LESSONS-LEARNED.md](LESSONS-LEARNED.md)

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Your contributions make this project better for everyone. Thank you for taking the time to contribute! 🎉

---

**Last Updated**: 2024-11-09
**Version**: 0.1.0
