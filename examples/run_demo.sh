#!/bin/bash
# Sheepy Demo — Bash launcher
# Run from the repo root: bash examples/run_demo.sh
set -e
cd "$(dirname "$0")/.."

echo "=== Installing sheepy in editable mode ==="
pip install -e ".[dev,vector,rag]"

echo "=== Running demo ==="
python examples/demo.py
