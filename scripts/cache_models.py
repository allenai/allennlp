#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.commands import DEFAULT_MODELS
from allennlp.common.file_utils import cached_path

value = os.environ.get('CACHE_MODELS')
if isinstance(value, str) and value.lower() == "true":
    urls = DEFAULT_MODELS.values()
    print("CACHE_MODELS is '%s'.  Downloading %i models." % (value, len(urls)))
    for i, url in enumerate(urls):
        print("Downloading model %i of %i from: %s" % (i, len(urls), url))
        print(cached_path(url))
else:
    print("CACHE_MODELS is '%s'.  Not caching models." % value)
