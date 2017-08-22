import os
import base64
import logging
from urllib.parse import urlparse

import requests
import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.allennlp'))
DATASET_CACHE = os.path.join(CACHE_ROOT, "datasets")

def url_to_filename(url: str) -> str:
    url_bytes = url.encode('utf-8')
    b64_bytes = base64.b64encode(url_bytes)
    return b64_bytes.decode('utf-8')

def filename_to_url(filename: str) -> str:
    filename_bytes = filename.encode('utf-8')
    url_bytes = base64.b64decode(filename_bytes)
    return url_bytes.decode('utf-8')

def cached_path(maybe_uri: str, cache_dir: str = DATASET_CACHE) -> str:
    """
    Given something that might be a URI (or might be a local path),
    determine which. If it's a URI, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure it exists and then return it.
    """
    parsed = urlparse(maybe_uri)

    if parsed.scheme in ('http', 'https'):
        return get_from_cache(maybe_uri, cache_dir)
    elif parsed.scheme == '' and os.path.exists(maybe_uri):
        return maybe_uri
    elif parsed.scheme == '':
        raise FileNotFoundError("file {} not found".format(maybe_uri))
    else:
        raise ValueError("unable to parse {} as a URL or as a local path".format(maybe_uri))


# TODO(joelgrus): do we want to do checksums or anything like that?
def get_from_cache(url: str, cache_dir: str = DATASET_CACHE) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, url_to_filename(url))

    if not os.path.exists(path):
        logger.info("%s not found in cache, downloading to %s", url, path)
        req = requests.get(url, stream=True)
        progress = tqdm.tqdm(unit="B", total=int(req.headers['Content-Length']))
        with open(path, 'wb') as output_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    output_file.write(chunk)

    return path
