#!/usr/bin/env bash
# Development environment setup script

set -e

echo "========================================="
echo "CI Platform Manager - Dev Setup"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✓ Virtual environment already exists at .venv"
else
    echo "→ Creating virtual environment..."
    make venv
    echo ""
fi

# Install dependencies
echo "→ Installing package with dev dependencies..."
make install
echo ""

# Verify installation
echo "→ Verifying installation..."
if [ -f ".venv/bin/pylint" ]; then
    echo "✓ pylint installed: $(.venv/bin/pylint --version | head -1)"
else
    echo "✗ pylint installation failed"
    exit 1
fi

if [ -f ".venv/bin/pytest" ]; then
    echo "✓ pytest installed: $(.venv/bin/pytest --version)"
else
    echo "✗ pytest installation failed"
    exit 1
fi

if [ -f ".venv/bin/black" ]; then
    echo "✓ black installed: $(.venv/bin/black --version)"
else
    echo "✗ black installation failed"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ Development environment ready!"
echo "========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Available commands:"
echo "  make pylint   - Run pylint on the codebase"
echo "  make lint     - Run all linters"
echo "  make test     - Run tests"
echo "  make format   - Format code with black"
echo ""
echo "Or run directly from venv:"
echo "  .venv/bin/pylint ci_platform_manager"
echo "  .venv/bin/pytest"
echo ""
