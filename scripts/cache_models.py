#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.commands.serve import DEFAULT_MODELS
from allennlp.common.file_utils import cached_path

value = os.environ.get('CACHE_MODELS', 'false')
if value.lower() == "true":
    models = DEFAULT_MODELS.items()
    print("CACHE_MODELS is '%s'.  Downloading %i models." % (value, len(models)))
    for i, (model, url) in enumerate(models):
        print("Downloading '%s' model from %s" % (model, url))
        print("Saved at %s" % cached_path(url))
else:
    print("CACHE_MODELS is '%s'.  Not caching models." % value)
