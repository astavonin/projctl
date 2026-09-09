# Virtual environment settings
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYLINT := $(VENV)/bin/pylint
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
FLAKE8 := $(VENV)/bin/flake8
MYPY := $(VENV)/bin/mypy

.PHONY: help venv install install-hooks test lint pylint format clean clean-venv

help:
	@echo "Available commands:"
	@echo "  make venv          - Create virtual environment"
	@echo "  make install       - Install package with dev dependencies in venv"
	@echo "  make install-hooks - Install the git pre-commit guard"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run all linters (pylint, flake8, mypy, shellcheck)"
	@echo "  make pylint        - Run pylint only"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make clean-venv    - Remove virtual environment"
	@echo ""
	@echo "First time setup:"
	@echo "  make venv && make install"

# Check if venv exists and is functional, recreate if not
$(VENV)/bin/pip:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"

venv: $(VENV)/bin/pip

# Install package and dependencies in venv
install: $(VENV)/bin/pip install-hooks
	@echo "Installing package with dev dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@echo "Installing CLI globally via pipx (editable)..."
	pipx install -e . --force
	@echo ""
	@echo "✓ Installation complete!"
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV)/bin/activate"

# Install the proprietary-identifier guard. .git/hooks is not versioned, so every
# clone needs this run once — `make install` depends on it so that happens by default.
# The denylist is seeded empty: its entries are themselves proprietary, so they live
# in .git/ where git cannot commit, push, or clone them.
install-hooks:
	@GIT_COMMON_DIR=$$(git rev-parse --git-common-dir 2>/dev/null) || \
		{ echo "Not a git working tree — nothing to install."; exit 0; }; \
	HOOKS="$$GIT_COMMON_DIR/hooks"; \
	mkdir -p "$$HOOKS"; \
	if [ -e "$$HOOKS/pre-commit" ] && ! grep -q 'projctl-managed-hook-shim' "$$HOOKS/pre-commit" 2>/dev/null; then \
		BAK="$$HOOKS/pre-commit.local.bak"; \
		if [ -e "$$BAK" ]; then \
			N=1; while [ -e "$$BAK.$$N" ]; do N=$$((N+1)); done; BAK="$$BAK.$$N"; \
		fi; \
		mv "$$HOOKS/pre-commit" "$$BAK"; \
		echo "! Preserved your existing pre-commit hook at $$BAK"; \
	fi; \
	printf '%s\n' \
		'#!/usr/bin/env bash' \
		'# projctl-managed-hook-shim — do not edit; edit scripts/pre-commit instead.' \
		'# Execs the tracked script so a pulled update takes effect with no reinstall;' \
		'# a copied hook would silently keep running the old ruleset.' \
		'# Fails CLOSED when the tracked script is unavailable: a checkout predating it,' \
		'# or a stray chmod -x, would otherwise disarm the guard with only a warning.' \
		'ROOT=$$(git rev-parse --show-toplevel) || {' \
		'    echo "pre-commit: not inside a git working tree — refusing to commit unchecked" >&2' \
		'    exit 1' \
		'}' \
		'HOOK="$$ROOT/scripts/pre-commit"' \
		'if [ ! -x "$$HOOK" ]; then' \
		'    echo "pre-commit: $$HOOK is missing or not executable — refusing to commit" >&2' \
		'    echo "  This branch may predate the guard. To commit anyway: git commit --no-verify" >&2' \
		'    exit 1' \
		'fi' \
		'exec "$$HOOK" "$$@"' \
		> "$$HOOKS/pre-commit"; \
	chmod +x "$$HOOKS/pre-commit"; \
	if [ ! -f "$$GIT_COMMON_DIR/proprietary-terms" ]; then \
		cp scripts/proprietary-terms.template "$$GIT_COMMON_DIR/proprietary-terms"; \
		chmod 600 "$$GIT_COMMON_DIR/proprietary-terms"; \
		echo "✓ Seeded $$GIT_COMMON_DIR/proprietary-terms (mode 600 — it holds the terms themselves)"; \
		echo "  It has no terms yet, so literal-term checking is OFF until you add some."; \
	fi; \
	echo "✓ pre-commit hook installed ($$HOOKS/pre-commit -> scripts/pre-commit)"

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
	$(FLAKE8) projctl
	@echo ""
	@echo "Running mypy..."
	$(MYPY) projctl
	@echo ""
	@echo "Running shellcheck on scripts/..."
	@if command -v shellcheck >/dev/null 2>&1; then \
		shellcheck -s bash scripts/pre-commit && echo "shellcheck: clean"; \
	else \
		echo "shellcheck not installed — skipping (install it to lint scripts/pre-commit)"; \
	fi

# Run pylint only
pylint: venv
	@if [ ! -f $(PYLINT) ]; then \
		echo "Pylint not installed. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Running pylint on projctl..."
	$(PYLINT) projctl

# Format code
format: venv
	@if [ ! -f $(BLACK) ]; then \
		echo "Black not installed. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Formatting with black..."
	$(BLACK) projctl tests

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
