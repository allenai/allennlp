#!/bin/bash

# Dispatches to allennlp train. Recovers if the serialization directory is
# found and is non-empty, trains from scratch otherwise.
#
# Usage:
# resumable_train.sh serialization_dir [train_arg ...]

serialization_dir=$1
shift

# If $serialization_dir exists and is non-empty we are resuming
if [ -d $serialization_dir ] && [ "$(ls -A $serialization_dir)" ]; then
    echo "Recovering state from $serialization_dir"
    allennlp train -r -s $serialization_dir $@
else
    echo "No recovery state found. Starting from scratch."
    allennlp train -s $serialization_dir $@
fi

