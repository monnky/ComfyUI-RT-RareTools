#!/usr/bin/env bash
# =============================================================================
# RT-RareTools Auto-Installer (Linux / macOS) - v0.4.1
# Automatically installs requirements and resolves matching llama-cpp-python CUDA/Metal wheel
# 100% English ASCII Only
# =============================================================================

set -e

# Find Python executable
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_EXE="$VIRTUAL_ENV/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXE="python3"
elif command -v python &>/dev/null; then
    PYTHON_EXE="python"
else
    echo "[ERROR] Python not found. Please ensure Python is in your PATH."
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "======================================================="
echo "RT-RareTools Auto-Installer (Linux / macOS) - v0.4.1"
echo "Using Python: $($PYTHON_EXE --version)"
echo "Target llama-cpp-python: v0.4.1"
echo "======================================================="

"$PYTHON_EXE" "$DIR/install.py" "$@"

echo "======================================================="
echo "Installation complete!"
echo "======================================================="
