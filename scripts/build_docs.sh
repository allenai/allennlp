#!/usr/bin/env bash

set -Eeuo pipefail

make clean
make build-all-api-docs
make build-docs
