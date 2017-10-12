#!/bin/bash

if [[ "$CACHE_MODELS" == "true" ]]; then
    python -c "from allennlp.commands import DEFAULT_MODELS
from allennlp.common.file_utils import cached_path
for url in DEFAULT_MODELS.values():
    print(cached_path(url))"
fi
