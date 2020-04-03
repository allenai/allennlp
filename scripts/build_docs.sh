#!/usr/bin/env bash
set -Eeuo pipefail

# Detect sed
SED=sed
if [[ $(uname) == 'Darwin' ]]; then
  if which gsed > /dev/null; then
    SED=gsed
  else
    echo "On MacOS, please install GNU sed with 'brew install gnu-sed'." >&2
    exit 1
  fi
fi

cp README.md docs/README.md
# Alter the relative path of the README image for the docs.
$SED -i '1s/docs/./' docs/README.md
cp LICENSE docs/LICENSE.md
cp ROADMAP.md docs/ROADMAP.md
cp CONTRIBUTING.md docs/CONTRIBUTING.md
cp mkdocs-skeleton.yml mkdocs.yml
python scripts/build_docs.py

mkdocs build
