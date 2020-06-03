"""
Utilities for working with the local dataset cache.
"""

import glob
import os
import logging
import shutil
import tempfile
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple, Union, IO, Callable, Set, List
from hashlib import sha256
from functools import wraps

import boto3
import botocore
from botocore.exceptions import ClientError, EndpointConnectionError
from filelock import FileLock
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError
from requests.packages.urllib3.util.retry import Retry

from allennlp.common.tqdm import Tqdm

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(os.getenv("ALLENNLP_CACHE_ROOT", Path.home() / ".allennlp"))
CACHE_DIRECTORY = str(CACHE_ROOT / "cache")
DEPRECATED_CACHE_DIRECTORY = str(CACHE_ROOT / "datasets")

# This variable was deprecated in 0.7.2 since we use a single folder for caching
# all types of files (datasets, models, etc.)
DATASET_CACHE = CACHE_DIRECTORY

# Warn if the user is still using the deprecated cache directory.
if os.path.exists(DEPRECATED_CACHE_DIRECTORY):
    logger = logging.getLogger(__name__)
    logger.warning(
        f"Deprecated cache directory found ({DEPRECATED_CACHE_DIRECTORY}).  "
        f"Please remove this directory from your system to free up space."
    )


def url_to_filename(url: str, etag: str = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def filename_to_url(filename: str, cache_dir: str = None) -> Tuple[str, str]:
    """
    Return the url and etag (which may be `None`) stored for `filename`.
    Raise `FileNotFoundError` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError("file {} not found".format(meta_path))

    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def cached_path(url_or_filename: Union[str, Path], cache_dir: str = None) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    url_or_filename = os.path.expanduser(url_or_filename)
    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https", "s3"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def is_url_or_existing_file(url_or_filename: Union[str, Path, None]) -> bool:
    """
    Given something that might be a URL (or might be a local path),
    determine check if it's url or an existing file path.
    """
    if url_or_filename is None:
        return False
    url_or_filename = os.path.expanduser(str(url_or_filename))
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https", "s3") or os.path.exists(url_or_filename)


def _split_s3_path(url: str) -> Tuple[str, str]:
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def _s3_request(func: Callable):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise FileNotFoundError("file {} not found".format(url))
            else:
                raise

    return wrapper


def _get_s3_resource():
    session = boto3.session.Session()
    if session.get_credentials() is None:
        # Use unsigned requests.
        s3_resource = session.resource(
            "s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED)
        )
    else:
        s3_resource = session.resource("s3")
    return s3_resource


@_s3_request
def _s3_etag(url: str) -> Optional[str]:
    """Check ETag on S3 object."""
    s3_resource = _get_s3_resource()
    bucket_name, s3_path = _split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@_s3_request
def _s3_get(url: str, temp_file: IO) -> None:
    """Pull a file directly from S3."""
    s3_resource = _get_s3_resource()
    bucket_name, s3_path = _split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def _session_with_backoff() -> requests.Session:
    """
    We ran into an issue where http requests to s3 were timing out,
    possibly because we were making too many requests too quickly.
    This helper function returns a requests session that has retry-with-backoff
    built in. See
    <https://stackoverflow.com/questions/23267409/how-to-implement-retry-mechanism-into-python-requests-library>.
    """
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    return session


def _http_etag(url: str) -> Optional[str]:
    with _session_with_backoff() as session:
        response = session.head(url, allow_redirects=True)
    if response.status_code != 200:
        raise IOError(
            "HEAD request failed for url {} with status code {}".format(url, response.status_code)
        )
    return response.headers.get("ETag")


def _http_get(url: str, temp_file: IO) -> None:
    with _session_with_backoff() as session:
        req = session.get(url, stream=True)
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = Tqdm.tqdm(unit="B", total=total)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)
        progress.close()


def _find_latest_cached(url: str, cache_dir: str) -> Optional[str]:
    filename = url_to_filename(url)
    cache_path = os.path.join(cache_dir, filename)
    candidates: List[Tuple[str, float]] = []
    for path in glob.glob(cache_path + "*"):
        if path.endswith(".json"):
            continue
        mtime = os.path.getmtime(path)
        candidates.append((path, mtime))
    # Sort candidates by modification time, neweste first.
    candidates.sort(key=lambda x: x[1], reverse=True)
    if candidates:
        return candidates[0][0]
    return None


# TODO(joelgrus): do we want to do checksums or anything like that?
def get_from_cache(url: str, cache_dir: str = None) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    os.makedirs(cache_dir, exist_ok=True)

    # Get eTag to add to filename, if it exists.
    try:
        if url.startswith("s3://"):
            etag = _s3_etag(url)
        else:
            etag = _http_etag(url)
    except (ConnectionError, EndpointConnectionError):
        # We might be offline, in which case we don't want to throw an error
        # just yet. Instead, we'll try to use the latest cached version of the
        # target resource, if it exists. We'll only throw an exception if we
        # haven't cached the resource at all yet.
        logger.warning(
            "Connection error occured while trying to fetch ETag for %s. "
            "Will attempt to use latest cached version of resource",
            url,
        )
        latest_cached = _find_latest_cached(url, cache_dir)
        if latest_cached:
            logger.info(
                "ETag request failed with connection error, using latest cached "
                "version of %s: %s",
                url,
                latest_cached,
            )
            return latest_cached
        else:
            logger.error(
                "Connection failed while trying to fetch ETag, "
                "and no cached version of %s could be found",
                url,
            )
            raise

    filename = url_to_filename(url, etag)

    # Get cache path to put the file.
    cache_path = os.path.join(cache_dir, filename)

    # Multiple processes may be trying to cache the same file at once, so we need
    # to be a little careful to avoid race conditions. We do this using a lock file.
    # Only one process can own this lock file at a time, and a process will block
    # on the call to `lock.acquire()` until the process currently holding the lock
    # releases it.
    logger.info("checking cache for %s at %s", url, cache_path)
    logger.info("waiting to acquire lock on %s", cache_path)
    with FileLock(cache_path + ".lock"):
        if os.path.exists(cache_path):
            logger.info("cache of %s is up-to-date", url)
        else:
            # Download to temporary file, then copy to cache dir once finished.
            # Otherwise you get corrupt cache entries if the download gets interrupted.
            with tempfile.NamedTemporaryFile() as temp_file:
                logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

                # GET file object
                if url.startswith("s3://"):
                    _s3_get(url, temp_file)
                else:
                    _http_get(url, temp_file)

                # we are copying the file before closing it, so flush to avoid truncation
                temp_file.flush()
                # shutil.copyfileobj() starts at the current position, so go to the start
                temp_file.seek(0)

                logger.info("copying %s to cache at %s", temp_file.name, cache_path)
                with open(cache_path, "wb") as cache_file:
                    shutil.copyfileobj(temp_file, cache_file)  # type: ignore

                logger.info("creating metadata file for %s", cache_path)
                meta = {"url": url, "etag": etag}
                meta_path = cache_path + ".json"
                with open(meta_path, "w") as meta_file:
                    json.dump(meta, meta_file)

                logger.info("removing temp file %s", temp_file.name)

    return cache_path


def read_set_from_file(filename: str) -> Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, "r") as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def get_file_extension(path: str, dot=True, lower: bool = True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


def open_compressed(
    filename: Union[str, Path], mode: str = "rt", encoding: Optional[str] = "UTF-8", **kwargs
):
    if isinstance(filename, Path):
        filename = str(filename)
    open_fn: Callable = open

    if filename.endswith(".gz"):
        import gzip

        open_fn = gzip.open
    elif filename.endswith(".bz2"):
        import bz2

        open_fn = bz2.open
    return open_fn(filename, mode=mode, encoding=encoding, **kwargs)
