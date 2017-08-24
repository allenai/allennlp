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
    """
    Converts a url into a filename in a reversible way.
    """
    url_bytes = url.encode('utf-8')
    b64_bytes = base64.b64encode(url_bytes)
    return b64_bytes.decode('utf-8')

def filename_to_url(filename: str) -> str:
    """
    Recovers the the url from the encoded filename.
    """
    filename_bytes = filename.encode('utf-8')
    url_bytes = base64.b64decode(filename_bytes)
    return url_bytes.decode('utf-8')

def cached_path(url_or_filename: str, cache_dir: str = DATASET_CACHE) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif parsed.scheme == '' and os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


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

        # GET file object
        req = requests.get(url, stream=True)
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm.tqdm(unit="B", total=total)
        with open(path, 'wb') as output_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    output_file.write(chunk)

    return path
