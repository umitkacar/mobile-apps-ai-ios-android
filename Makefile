.PHONY: help install dev-install clean test test-cov lint format check pre-commit build docs serve-docs

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)Mobile AI - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install          - Install package in production mode"
	@echo "  make dev-install      - Install package with dev dependencies"
	@echo "  make pre-commit-setup - Setup pre-commit hooks"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make test            - Run tests"
	@echo "  make test-cov        - Run tests with coverage"
	@echo "  make test-fast       - Run tests in parallel"
	@echo "  make lint            - Run linting checks"
	@echo "  make format          - Format code with black and ruff"
	@echo "  make type-check      - Run mypy type checking"
	@echo "  make check           - Run all checks (lint + format + type + test)"
	@echo ""
	@echo "$(GREEN)Cleanup:$(NC)"
	@echo "  make clean           - Remove build artifacts and cache"
	@echo "  make clean-all       - Deep clean (including venv)"
	@echo ""
	@echo "$(GREEN)Build & Deploy:$(NC)"
	@echo "  make build           - Build package distributions"
	@echo "  make docs            - Build documentation"
	@echo "  make serve-docs      - Serve documentation locally"
	@echo ""
	@echo "$(GREEN)CI/CD:$(NC)"
	@echo "  make ci              - Run full CI pipeline locally"
	@echo "  make pre-commit      - Run pre-commit hooks"

# ==================== Installation ====================

install:
	@echo "$(BLUE)Installing package...$(NC)"
	pip install -e .

dev-install:
	@echo "$(BLUE)Installing dev dependencies...$(NC)"
	pip install -e ".[dev]"
	@echo "$(GREEN)✓ Dev dependencies installed$(NC)"

pre-commit-setup:
	@echo "$(BLUE)Setting up pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

# ==================== Testing ====================

test:
	@echo "$(BLUE)Running tests...$(NC)"
	hatch run test

test-cov:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	hatch run test-cov
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-fast:
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	hatch run test-fast

test-verbose:
	@echo "$(BLUE)Running tests (verbose)...$(NC)"
	hatch run test-verbose

test-unit:
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest -m "unit" tests/

test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest -m "integration" tests/

test-ios:
	@echo "$(BLUE)Running iOS tests...$(NC)"
	pytest -m "ios" tests/

test-android:
	@echo "$(BLUE)Running Android tests...$(NC)"
	pytest -m "android" tests/

# ==================== Code Quality ====================

lint:
	@echo "$(BLUE)Running linting checks...$(NC)"
	hatch run lint

lint-fix:
	@echo "$(BLUE)Fixing linting issues...$(NC)"
	hatch run lint-fix
	@echo "$(GREEN)✓ Linting issues fixed$(NC)"

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	hatch run format
	hatch run lint-fix
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check:
	@echo "$(BLUE)Checking code formatting...$(NC)"
	hatch run format-check

type-check:
	@echo "$(BLUE)Running type checking...$(NC)"
	hatch run type-check

check: lint format-check type-check test-cov
	@echo "$(GREEN)✓ All checks passed!$(NC)"

# ==================== Pre-commit ====================

pre-commit:
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	hatch run pre-commit-run

pre-commit-update:
	@echo "$(BLUE)Updating pre-commit hooks...$(NC)"
	pre-commit autoupdate

# ==================== Build ====================

build:
	@echo "$(BLUE)Building package...$(NC)"
	python -m build
	@echo "$(GREEN)✓ Package built in dist/$(NC)"

build-check:
	@echo "$(BLUE)Checking built package...$(NC)"
	twine check dist/*

# ==================== Documentation ====================

docs:
	@echo "$(BLUE)Building documentation...$(NC)"
	hatch run docs:build
	@echo "$(GREEN)✓ Documentation built in site/$(NC)"

serve-docs:
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	hatch run docs:serve

# ==================== Cleanup ====================

clean:
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-all: clean
	@echo "$(BLUE)Deep cleaning...$(NC)"
	rm -rf .venv/
	rm -rf venv/
	rm -rf .hatch/
	@echo "$(GREEN)✓ Deep cleaned$(NC)"

# ==================== CI/CD ====================

ci: clean
	@echo "$(BLUE)Running full CI pipeline...$(NC)"
	@echo "$(YELLOW)1. Linting...$(NC)"
	@make lint
	@echo "$(YELLOW)2. Formatting check...$(NC)"
	@make format-check
	@echo "$(YELLOW)3. Type checking...$(NC)"
	@make type-check
	@echo "$(YELLOW)4. Running tests...$(NC)"
	@make test-cov
	@echo "$(YELLOW)5. Building package...$(NC)"
	@make build
	@echo "$(GREEN)✓ CI pipeline completed successfully!$(NC)"

# ==================== Development Helpers ====================

shell:
	@echo "$(BLUE)Starting development shell...$(NC)"
	hatch shell

jupyter:
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	jupyter notebook

coverage-report:
	@echo "$(BLUE)Generating coverage report...$(NC)"
	coverage report -m
	coverage html
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

benchmark:
	@echo "$(BLUE)Running benchmarks...$(NC)"
	pytest tests/ -v --benchmark-only

security:
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r src/ -c pyproject.toml
	safety check

# ==================== Version Management ====================

version:
	@echo "$(BLUE)Current version:$(NC)"
	@hatch version

version-bump-patch:
	@echo "$(BLUE)Bumping patch version...$(NC)"
	hatch version patch

version-bump-minor:
	@echo "$(BLUE)Bumping minor version...$(NC)"
	hatch version minor

version-bump-major:
	@echo "$(BLUE)Bumping major version...$(NC)"
	hatch version major

# ==================== Quick Commands ====================

# Quick test-driven development cycle
tdd:
	@echo "$(BLUE)Starting TDD mode (watch for changes)...$(NC)"
	pytest-watch tests/

# Run tests and open coverage report
cov: test-cov
	@echo "$(BLUE)Opening coverage report...$(NC)"
	open htmlcov/index.html || xdg-open htmlcov/index.html

# Quick format and test
ft: format test
	@echo "$(GREEN)✓ Format and test complete!$(NC)"
