#!/bin/bash
# Clean script - removes all build artifacts and cache files

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧹 Cleaning build artifacts and cache...${NC}"

# Python cache
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

# Build artifacts
echo "Removing build artifacts..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf .eggs/

# Test and coverage
echo "Removing test artifacts..."
rm -rf .pytest_cache/
rm -rf .tox/
rm -rf htmlcov/
rm -rf .coverage
rm -rf coverage.xml
rm -rf .coverage.*

# Linting and type checking
echo "Removing linting cache..."
rm -rf .mypy_cache/
rm -rf .ruff_cache/
rm -rf .pytype/

# Documentation
echo "Removing documentation build..."
rm -rf site/
rm -rf docs/_build/

# Jupyter
echo "Removing Jupyter artifacts..."
find . -type f -name "*.ipynb_checkpoints" -delete
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# macOS
echo "Removing macOS files..."
find . -type f -name ".DS_Store" -delete

# IDE
echo "Removing IDE files..."
rm -rf .idea/
rm -rf .vscode/
rm -rf *.swp
rm -rf *.swo
rm -rf *~

echo -e "${GREEN}✓ Clean complete!${NC}"
