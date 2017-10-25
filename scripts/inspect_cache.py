import os
import base64
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.common.file_utils import filename_to_url
from allennlp.common.file_utils import DATASET_CACHE

try:
    cached_files = os.listdir(DATASET_CACHE)
    if not cached_files:
        print('No cached datasets found.')

    for filename in cached_files:
        url, etag = filename_to_url(filename)
        print('Filename: %s' % filename)
        print('Url: %s' % url)
        print('ETag: %s' % etag)
        print()
except FileNotFoundError:
    print('No cached datasets found.')    
