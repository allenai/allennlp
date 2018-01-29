#!/usr/bin/env bash
# Run type checking over the python code.

set -e
./scripts/verify.py --checks mypy
echo -e "mypy checks passed\n"
