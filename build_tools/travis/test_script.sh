#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

python --version

export PYTHONHASHSEED=2157

run_tests() {
    pytest -v --cov=allennlp --durations=20
}

if [[ "$RUN_PYLINT" == "true" ]]; then
    source scripts/pylint.sh
fi

if [[ "$RUN_MYPY" == "true" ]]; then
    source scripts/mypy.sh
fi

if [[ "$RUN_TESTS" == "true" ]]; then
    run_tests
fi

if [[ "$BUILD_DOCS" == "true" ]]; then
  cd doc
  make html-strict
  cd ..
fi

if [[ "$COVERAGE" == "true" ]]; then
    # Ignore codecov failures as the codecov server is not
    # very reliable but we don't want travis to report a failure
    # in the github UI just because the coverage report failed to
    # be published.
    codecov || echo "codecov upload failed"
fi
