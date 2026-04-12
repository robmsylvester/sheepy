#!/bin/bash
set -e
python -m pytest tests/ -vv --tb=short -m "not slow" "$@"
