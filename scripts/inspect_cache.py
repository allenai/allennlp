import os
import base64
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.common.file_utils import filename_to_url
from allennlp.common.file_utils import DATASET_CACHE

print(f"Looking for datasets in {DATASET_CACHE}...")
if not os.path.exists(DATASET_CACHE):
    print('Directory does not exist.')
    print('No cached datasets found.')

cached_files = os.listdir(DATASET_CACHE)

if not cached_files:
    print('Directory is empty.')
    print('No cached datasets found.')

for filename in cached_files:
    if not filename.endswith("json"):
        url, etag = filename_to_url(filename)
        print('Filename: %s' % filename)
        print('Url: %s' % url)
        print('ETag: %s' % etag)
        print()
