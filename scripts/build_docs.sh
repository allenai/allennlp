

set -e

#python scripts/build_docs.py
cp README.md docs/README.md
cp LICENSE docs/LICENSE.md
cp ROADMAP.md docs/ROADMAP.md
cp CONTRIBUTING.md docs/CONTRIBUTING.md

mkdocs build

