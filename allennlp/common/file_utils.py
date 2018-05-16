"""
Utilities for working with the local dataset cache.
"""

from typing import Tuple
import os
import base64
import logging
import shutil
import tempfile
from urllib.parse import urlparse
from contextlib import contextmanager
import sqlite3

import requests

from allennlp.common.tqdm import Tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CACHE_ROOT = os.getenv('ALLENNLP_CACHE_ROOT', os.path.expanduser(os.path.join('~', '.allennlp')))
DATASET_CACHE = os.path.join(CACHE_ROOT, "datasets")
INDEX_FILENAME = 'index.db'

def url_to_filename(url: str, etag: str = None) -> str:
    """
    Converts a url into a filename in a reversible way.
    If `etag` is specified, add it on the end, separated by a period
    (which necessarily won't appear in the base64-encoded filename).
    Get rid of the quotes in the etag, since Windows doesn't like them.
    """
    url_bytes = url.encode('utf-8')
    b64_bytes = base64.b64encode(url_bytes)
    decoded = b64_bytes.decode('utf-8')

    if etag:
        # Remove quotes from etag
        etag = etag.replace('"', '')
        return f"{decoded}.{etag}"
    else:
        return decoded

def filename_to_url(filename: str) -> Tuple[str, str]:
    """
    Recovers the the url from the encoded filename. Returns it and the ETag
    (which may be ``None``)
    """
    try:
        # If there is an etag, it's everything after the first period
        decoded, etag = filename.split(".", 1)
    except ValueError:
        # Otherwise, use None
        decoded, etag = filename, None

    filename_bytes = decoded.encode('utf-8')
    url_bytes = base64.b64decode(filename_bytes)
    return url_bytes.decode('utf-8'), etag

def cached_path(url_or_filename: str, cache_dir: str = None) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = DATASET_CACHE

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
def get_from_cache(url: str, cache_dir: str = None) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = DATASET_CACHE

    # make HEAD request to check ETag
    response = requests.head(url)
    if response.status_code != 200:
        raise IOError("HEAD request failed for url {}".format(url))

    # add ETag to filename if it exists
    etag = response.headers.get("ETag")

    with _open_cache_index(cache_dir) as cursor:
        filename = _url_to_filename(cursor, url, etag)

        # filename will be None if it isn't cached
        if filename is None:
            # Download to temporary file, then copy to cache dir once finished.
            # Otherwise you get corrupt cache entries if the download gets interrupted.
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)
            # GET file object
            req = requests.get(url, stream=True)
            content_length = req.headers.get('Content-Length')
            total = int(content_length) if content_length is not None else None
            progress = Tqdm.tqdm(unit="B", total=total)
            for chunk in req.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)
            progress.close()
            temp_file.close()

            # Download succeeded. Now open cache file.
            cache_file = tempfile.NamedTemporaryFile(dir=cache_dir, delete=False)
            cache_file.close()  # just need the name
            cache_path = cache_file.name

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            shutil.copyfile(temp_file.name, cache_path)
            filename = os.path.basename(cache_path)

            logger.info("adding (%s, %s, %s) to the cache index", url, etag, filename)
            _index_file(cursor, url, etag, filename)

            logger.info("removing temp file %s", temp_file.name)
            os.remove(temp_file.name)
        else:
            cache_path = os.path.join(cache_dir, filename)
            if not os.path.exists(cache_path):
                raise FileNotFoundError("indexed dataset does not exist in cache: {}".format(cache_path))

    return cache_path

@contextmanager
def _open_cache_index(cache_dir: str = None) -> sqlite3.Cursor:
    """
    Create (if necessary) and open the cache index database.
    """

    if cache_dir is None:
        cache_dir = DATASET_CACHE

    os.makedirs(cache_dir, exist_ok=True)

    database_path = os.path.join(cache_dir, INDEX_FILENAME)
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # initialize tables if database is new
    cursor.execute("CREATE TABLE IF NOT EXISTS datasets (url text, etag text, filename text)")
    connection.commit()

    # clean up database (remove rows if the file no longer exists)
    for row in cursor.execute("SELECT rowid, url, etag, filename FROM datasets"):
        if not os.path.isfile(os.path.join(cache_dir, row[3])):
            etag_string = ", etag=" + row[2] if row[2] is not None else ""
            logger.info("De-indexing missing dataset: {}{}".format(row[1], etag_string))
            cursor.execute("DELETE FROM datasets WHERE rowid=?", (row[0],))

    yield cursor

    connection.commit()
    connection.close()

def _url_to_filename(cursor: sqlite3.Cursor, url: str, etag: str = None) -> str:
    row = cursor.execute("SELECT filename FROM datasets WHERE url=? AND etag=?", (url, etag)).fetchone()
    filename = None if row is None else row[0]
    return filename

def _filename_to_url(cursor: sqlite3.Cursor, filename: str) -> Tuple[str, str]:
    row = cursor.execute("SELECT url, etag FROM datasets WHERE filename=?", (filename,)).fetchone()
    return row

def _index_file(cursor: sqlite3.Cursor, url: str, etag: str, filename: str):
    cursor.execute("INSERT INTO datasets(url, etag, filename) values (?, ?, ?)", (url, etag, filename))
