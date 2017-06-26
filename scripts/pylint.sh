#!/usr/bin/env bash
set -e
echo 'Starting pylint checks'
pylint -d locally-disabled,locally-enabled -f colorized allennlp tests
echo -e "pylint checks passed\n"
