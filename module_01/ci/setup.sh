#!/usr/bin/env bash

set -e  # Exit immediately on error

# Go to the parent directory of the script (module_01/)
cd "$(dirname "$0")/.."

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

# Ensure requirements.txt exists
if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "❌ $REQUIREMENTS not found in $(pwd)."
  exit 1
fi

echo "🔧 Creating virtual environment in $VENV_DIR ..."
uv venv "$VENV_DIR"

echo "🔄 Syncing dependencies from $REQUIREMENTS ..."
uv pip sync "$REQUIREMENTS"

echo "✅ Environment setup complete."
echo "👉 To activate the environment, run:"
echo "   source $(pwd)/$VENV_DIR/bin/activate"
