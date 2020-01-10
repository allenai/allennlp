#!/usr/bin/env bash
set -e

python scripts/build_docs.py
cp README.md docs/README.md
# Alter the relative path of the README image for the docs.
sed -i '' '1s/docs/./' docs/README.md
cp LICENSE docs/LICENSE.md
cp ROADMAP.md docs/ROADMAP.md
cp CONTRIBUTING.md docs/CONTRIBUTING.md
