#!/usr/bin/env bash
set -e
echo 'Starting mypy checks'
mypy allennlp --ignore-missing-imports
echo -e "mypy checks passed\n"
