#!/usr/bin/env bash
set -x
set -e

MAKE_TARGET=html-strict

source activate testenv

# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make $MAKE_TARGET 2>&1 | tee ~/log.txt

echo "Finished building docs."
echo "Artifacts in $CIRCLE_ARTIFACTS"
