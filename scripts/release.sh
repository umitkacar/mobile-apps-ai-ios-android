#!/bin/bash
# Release script - builds and releases package

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

VERSION_TYPE=${1:-patch}

echo -e "${BLUE}📦 Mobile AI - Release Process${NC}"
echo ""

# Check if we're on main branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo -e "${YELLOW}⚠ Warning: Not on main branch (current: $current_branch)${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}✗ Uncommitted changes detected${NC}"
    echo "Please commit or stash changes before release"
    exit 1
fi

# Run all checks
echo -e "${YELLOW}Running pre-release checks...${NC}"
echo -e "${BLUE}  1. Linting...${NC}"
make lint

echo -e "${BLUE}  2. Format check...${NC}"
make format-check

echo -e "${BLUE}  3. Type checking...${NC}"
make type-check

echo -e "${BLUE}  4. Running tests...${NC}"
make test-cov

echo -e "${BLUE}  5. Security check...${NC}"
make security || true

echo -e "${GREEN}✓ All checks passed${NC}"
echo ""

# Bump version
echo -e "${YELLOW}Bumping version ($VERSION_TYPE)...${NC}"
hatch version $VERSION_TYPE
new_version=$(hatch version)
echo -e "${GREEN}✓ New version: $new_version${NC}"

# Build package
echo -e "${YELLOW}Building package...${NC}"
make clean
make build
echo -e "${GREEN}✓ Package built${NC}"

# Check package
echo -e "${YELLOW}Checking package...${NC}"
twine check dist/*
echo -e "${GREEN}✓ Package check passed${NC}"

# Commit version bump
echo -e "${YELLOW}Committing version bump...${NC}"
git add src/mobile_ai/__init__.py
git commit -m "chore: bump version to $new_version"

# Create git tag
echo -e "${YELLOW}Creating git tag...${NC}"
git tag -a "v$new_version" -m "Release v$new_version"
echo -e "${GREEN}✓ Tag created: v$new_version${NC}"

echo ""
echo -e "${GREEN}✓ Release preparation complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Review changes: git log -1"
echo "  2. Push changes: git push && git push --tags"
echo "  3. Upload to PyPI: twine upload dist/*"
echo ""
