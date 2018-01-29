#!/usr/bin/env bash
# Run our linter over the python code.

set -e
./scripts/verify.py --checks pylint
echo -e "pylint checks passed\n"
