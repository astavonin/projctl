# Virtual environment settings
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYLINT := $(VENV)/bin/pylint
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
FLAKE8 := $(VENV)/bin/flake8
MYPY := $(VENV)/bin/mypy

.PHONY: help venv install test lint pylint format clean clean-venv

help:
	@echo "Available commands:"
	@echo "  make venv       - Create virtual environment"
	@echo "  make install    - Install package with dev dependencies in venv"
	@echo "  make test       - Run tests with coverage"
	@echo "  make lint       - Run all linters (pylint, flake8, mypy)"
	@echo "  make pylint     - Run pylint only"
	@echo "  make format     - Format code with black"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make clean-venv - Remove virtual environment"
	@echo ""
	@echo "First time setup:"
	@echo "  make venv && make install"

# Check if venv exists, create if not
$(VENV)/bin/activate:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"

venv: $(VENV)/bin/activate

# Install package and dependencies in venv
install: venv
	@echo "Installing package with dev dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "✓ Installation complete!"
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV)/bin/activate"

# Run tests
test: venv
	@if [ ! -f $(PYTEST) ]; then \
		echo "Dependencies not installed. Run 'make install' first."; \
		exit 1; \
	fi
	$(PYTEST)

# Run all linters
lint: pylint
	@echo "Running flake8..."
	$(FLAKE8) ci_platform_manager || true
	@echo ""
	@echo "Running mypy..."
	$(MYPY) ci_platform_manager || true

# Run pylint only
pylint: venv
	@if [ ! -f $(PYLINT) ]; then \
		echo "Pylint not installed. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Running pylint on ci_platform_manager..."
	$(PYLINT) ci_platform_manager

# Format code
format: venv
	@if [ ! -f $(BLACK) ]; then \
		echo "Black not installed. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Formatting with black..."
	$(BLACK) ci_platform_manager tests

# Clean build artifacts
clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

# Remove virtual environment
clean-venv:
	rm -rf $(VENV)
	@echo "Virtual environment removed."
