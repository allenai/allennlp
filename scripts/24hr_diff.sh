#!/usr/bin/env bash

# A small script which checks if there have been any commits in the past 24 hours.

if [[ $(git whatchanged --since 'one day ago') ]]; then
  exit 0
fi
exit 1
