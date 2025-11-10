#!/bin/bash
# Setup script for mobile-ai development environment

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Mobile AI - Development Environment Setup${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version | cut -d' ' -f2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}✗ Python 3.9+ required. Found: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install package with dev dependencies
echo -e "${YELLOW}Installing package with dev dependencies...${NC}"
pip install -e ".[dev]"
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Setup pre-commit hooks
echo -e "${YELLOW}Setting up pre-commit hooks...${NC}"
pre-commit install
pre-commit install --hook-type commit-msg
echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"

# Create secrets baseline for detect-secrets
if [ ! -f ".secrets.baseline" ]; then
    echo -e "${YELLOW}Creating secrets baseline...${NC}"
    detect-secrets scan > .secrets.baseline || true
    echo -e "${GREEN}✓ Secrets baseline created${NC}"
fi

# Run initial checks
echo -e "${YELLOW}Running initial checks...${NC}"
echo -e "${BLUE}  - Linting...${NC}"
ruff check src tests || true

echo -e "${BLUE}  - Formatting...${NC}"
black --check src tests || true

echo -e "${BLUE}  - Type checking...${NC}"
mypy src tests || true

echo ""
echo -e "${GREEN}✓ Development environment setup complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run tests: make test"
echo "  3. Check all: make check"
echo "  4. See all commands: make help"
echo ""
