"""
Utilities for working with the local dataset cache.
"""

import glob
import os
import logging
import tempfile
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import timedelta
from fnmatch import fnmatch
from os import PathLike
from urllib.parse import urlparse
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    Union,
    IO,
    Callable,
    Set,
    List,
    Iterator,
    Iterable,
    Dict,
    NamedTuple,
)
from hashlib import sha256
from functools import wraps
from zipfile import ZipFile, is_zipfile
import tarfile
import shutil
import time

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
    logger.warning(
        f"Deprecated cache directory found ({DEPRECATED_CACHE_DIRECTORY}).  "
        f"Please remove this directory from your system to free up space."
    )


def _resource_to_filename(resource: str, etag: str = None) -> str:
    """
    Convert a `resource` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the resources's, delimited
    by a period.
    """
    resource_bytes = resource.encode("utf-8")
    resource_hash = sha256(resource_bytes)
    filename = resource_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def filename_to_url(filename: str, cache_dir: Union[str, Path] = None) -> Tuple[str, str]:
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


def cached_path(
    url_or_filename: Union[str, PathLike],
    cache_dir: Union[str, Path] = None,
    extract_archive: bool = False,
    force_extract: bool = False,
) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.

    # Parameters

    url_or_filename : `Union[str, Path]`
        A URL or local file to parse and possibly download.

    cache_dir : `Union[str, Path]`, optional (default = `None`)
        The directory to cache downloads.

    extract_archive : `bool`, optional (default = `False`)
        If `True`, then zip or tar.gz archives will be automatically extracted.
        In which case the directory is returned.

    force_extract : `bool`, optional (default = `False`)
        If `True` and the file is an archive file, it will be extracted regardless
        of whether or not the extracted directory already exists.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if not isinstance(url_or_filename, str):
        url_or_filename = str(url_or_filename)

    file_path: str

    # If we're using the /a/b/foo.zip!c/d/file.txt syntax, handle it here.
    exclamation_index = url_or_filename.find("!")
    if extract_archive and exclamation_index >= 0:
        archive_path = url_or_filename[:exclamation_index]
        file_name = url_or_filename[exclamation_index + 1 :]

        # Call 'cached_path' recursively now to get the local path to the archive itself.
        cached_archive_path = cached_path(archive_path, cache_dir, True, force_extract)
        if not os.path.isdir(cached_archive_path):
            raise ValueError(
                f"{url_or_filename} uses the ! syntax, but does not specify an archive file."
            )

        # Now return the full path to the desired file within the extracted archive,
        # provided it exists.
        file_path = os.path.join(cached_archive_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file {file_name} not found within {archive_path}")

        return file_path

    parsed = urlparse(url_or_filename)

    extraction_path: Optional[str] = None

    if parsed.scheme in ("http", "https", "s3"):
        # URL, so get it from the cache (downloading if necessary)
        file_path = get_from_cache(url_or_filename, cache_dir)

        if extract_archive and (is_zipfile(file_path) or tarfile.is_tarfile(file_path)):
            # This is the path the file should be extracted to.
            # For example ~/.allennlp/cache/234234.21341 -> ~/.allennlp/cache/234234.21341-extracted
            extraction_path = file_path + "-extracted"

    else:
        url_or_filename = os.path.expanduser(url_or_filename)

        if os.path.exists(url_or_filename):
            # File, and it exists.
            file_path = url_or_filename
            # Normalize the path.
            url_or_filename = os.path.abspath(url_or_filename)

            if extract_archive and (is_zipfile(file_path) or tarfile.is_tarfile(file_path)):
                # We'll use a unique directory within the cache to root to extract the archive to.
                # The name of the directoy is a hash of the resource file path and it's modification
                # time. That way, if the file changes, we'll know when to extract it again.
                extraction_name = (
                    _resource_to_filename(url_or_filename, str(os.path.getmtime(file_path)))
                    + "-extracted"
                )
                extraction_path = os.path.join(cache_dir, extraction_name)

        elif parsed.scheme == "":
            # File, but it doesn't exist.
            raise FileNotFoundError(f"file {url_or_filename} not found")

        else:
            # Something unknown
            raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if extraction_path is not None:
        # If the extracted directory already exists (and is non-empty), then no
        # need to extract again unless `force_extract=True`.
        if os.path.isdir(extraction_path) and os.listdir(extraction_path) and not force_extract:
            return extraction_path

        # Extract it.
        with FileLock(extraction_path + ".lock"):
            logger.info("Extracting %s to %s", url_or_filename, extraction_path)
            shutil.rmtree(extraction_path, ignore_errors=True)

            # We extract first to a temporary directory in case something goes wrong
            # during the extraction process so we don't end up with a corrupted cache.
            tmp_extraction_dir = tempfile.mkdtemp(dir=os.path.split(extraction_path)[0])
            try:
                if is_zipfile(file_path):
                    with ZipFile(file_path, "r") as zip_file:
                        zip_file.extractall(tmp_extraction_dir)
                        zip_file.close()
                else:
                    tar_file = tarfile.open(file_path)
                    tar_file.extractall(tmp_extraction_dir)
                    tar_file.close()
                # Extraction was successful, rename temp directory to final
                # cache directory and dump the meta data.
                os.replace(tmp_extraction_dir, extraction_path)
                meta = _Meta(
                    resource=url_or_filename,
                    cached_path=extraction_path,
                    creation_time=time.time(),
                    extraction_dir=True,
                    size=_get_resource_size(extraction_path),
                )
                meta.to_file()
            finally:
                shutil.rmtree(tmp_extraction_dir, ignore_errors=True)

        return extraction_path

    return file_path


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
        progress = Tqdm.tqdm(unit="B", total=total, desc="downloading")
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)
        progress.close()


def _find_latest_cached(url: str, cache_dir: Union[str, Path]) -> Optional[str]:
    filename = _resource_to_filename(url)
    cache_path = os.path.join(cache_dir, filename)
    candidates: List[Tuple[str, float]] = []
    for path in glob.glob(cache_path + "*"):
        if path.endswith(".json") or path.endswith("-extracted") or path.endswith(".lock"):
            continue
        mtime = os.path.getmtime(path)
        candidates.append((path, mtime))
    # Sort candidates by modification time, newest first.
    candidates.sort(key=lambda x: x[1], reverse=True)
    if candidates:
        return candidates[0][0]
    return None


class CacheFile:
    """
    This is a context manager that makes robust caching easier.

    On `__enter__`, an IO handle to a temporarily file is returned, which can
    be treated as if it's the actual cache file.

    On `__exit__`, the temporarily file is renamed to the cache file. If anything
    goes wrong while writing to the temporary file, it will be removed.
    """

    def __init__(
        self, cache_filename: Union[Path, str], mode: str = "w+b", suffix: str = ".tmp"
    ) -> None:
        self.cache_filename = (
            cache_filename if isinstance(cache_filename, Path) else Path(cache_filename)
        )
        self.cache_directory = os.path.dirname(self.cache_filename)
        self.mode = mode
        self.temp_file = tempfile.NamedTemporaryFile(
            self.mode, dir=self.cache_directory, delete=False, suffix=suffix
        )

    def __enter__(self):
        return self.temp_file

    def __exit__(self, exc_type, exc_value, traceback):
        self.temp_file.close()
        if exc_value is None:
            # Success.
            logger.debug(
                "Renaming temp file %s to cache at %s", self.temp_file.name, self.cache_filename
            )
            # Rename the temp file to the actual cache filename.
            os.replace(self.temp_file.name, self.cache_filename)
            return True
        # Something went wrong, remove the temp file.
        logger.debug("removing temp file %s", self.temp_file.name)
        os.remove(self.temp_file.name)
        return False


@dataclass
class _Meta:
    """
    Any resource that is downloaded to - or extracted in - the cache directory will
    have a meta JSON file written next to it, which corresponds to an instance
    of this class.

    In older versions of AllenNLP, this meta document just had two fields: 'url' and
    'etag'. The 'url' field is now the more general 'resource' field, but these old
    meta files are still compatible when a `_Meta` is instantiated with the `.from_path()`
    class method.
    """

    resource: str
    """
    URL or normalized path to the resource.
    """

    cached_path: str
    """
    Path to the corresponding cached version of the resource.
    """

    creation_time: float
    """
    The unix timestamp of when the corresponding resource was cached or extracted.
    """

    size: int = 0
    """
    The size of the corresponding resource, in bytes.
    """

    etag: Optional[str] = None
    """
    Optional ETag associated with the current cached version of the resource.
    """

    extraction_dir: bool = False
    """
    Does this meta corresponded to an extraction directory?
    """

    def to_file(self) -> None:
        with open(self.cached_path + ".json", "w") as meta_file:
            json.dump(asdict(self), meta_file)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "_Meta":
        path = str(path)
        with open(path) as meta_file:
            data = json.load(meta_file)
            # For backwards compat:
            if "resource" not in data:
                data["resource"] = data.pop("url")
            if "creation_time" not in data:
                data["creation_time"] = os.path.getmtime(path[:-5])
            if "extraction_dir" not in data and path.endswith("-extracted.json"):
                data["extraction_dir"] = True
            if "cached_path" not in data:
                data["cached_path"] = path[:-5]
            if "size" not in data:
                data["size"] = _get_resource_size(data["cached_path"])
        return cls(**data)


# TODO(joelgrus): do we want to do checksums or anything like that?
def get_from_cache(url: str, cache_dir: Union[str, Path] = None) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

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
            "Connection error occurred while trying to fetch ETag for %s. "
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
    except OSError:
        # OSError may be triggered if we were unable to fetch the eTag.
        # If this is the case, try to proceed without eTag check.
        etag = None

    filename = _resource_to_filename(url, etag)

    # Get cache path to put the file.
    cache_path = os.path.join(cache_dir, filename)

    # Multiple processes may be trying to cache the same file at once, so we need
    # to be a little careful to avoid race conditions. We do this using a lock file.
    # Only one process can own this lock file at a time, and a process will block
    # on the call to `lock.acquire()` until the process currently holding the lock
    # releases it.
    logger.debug("waiting to acquire lock on %s", cache_path)
    with FileLock(cache_path + ".lock"):
        if os.path.exists(cache_path):
            logger.info("cache of %s is up-to-date", url)
        else:
            with CacheFile(cache_path) as cache_file:
                logger.info("%s not found in cache, downloading to %s", url, cache_path)

                # GET file object
                if url.startswith("s3://"):
                    _s3_get(url, cache_file)
                else:
                    _http_get(url, cache_file)

            logger.debug("creating metadata file for %s", cache_path)
            meta = _Meta(
                resource=url,
                cached_path=cache_path,
                creation_time=time.time(),
                etag=etag,
                size=_get_resource_size(cache_path),
            )
            meta.to_file()

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


def text_lines_from_file(filename: Union[str, Path], strip_lines: bool = True) -> Iterator[str]:
    with open_compressed(filename, "rt", encoding="UTF-8", errors="replace") as p:
        if strip_lines:
            for line in p:
                yield line.strip()
        else:
            yield from p


def json_lines_from_file(filename: Union[str, Path]) -> Iterable[Union[list, dict]]:
    return (json.loads(line) for line in text_lines_from_file(filename))


def _get_resource_size(path: str) -> int:
    """
    Get the size of a file or directory.
    """
    if os.path.isfile(path):
        return os.path.getsize(path)
    inodes: Set[int] = set()
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link or the same as a file we've already accounted
            # for (this could happen with hard links).
            inode = os.stat(fp).st_ino
            if not os.path.islink(fp) and inode not in inodes:
                inodes.add(inode)
                total_size += os.path.getsize(fp)
    return total_size


class _CacheEntry(NamedTuple):
    regular_files: List[_Meta]
    extraction_dirs: List[_Meta]


def _find_entries(
    patterns: List[str] = None,
    cache_dir: Union[str, Path] = None,
) -> Tuple[int, Dict[str, _CacheEntry]]:
    """
    Find all cache entries, filtering ones that don't match any of the glob patterns given.

    Returns the total size of the matching entries and mapping or resource name to meta data.

    The values in the returned mapping are tuples because we seperate meta entries that
    correspond to extraction directories vs regular cache entries.
    """
    cache_dir = os.path.expanduser(cache_dir or CACHE_DIRECTORY)

    total_size: int = 0
    cache_entries: Dict[str, _CacheEntry] = defaultdict(lambda: _CacheEntry([], []))
    for meta_path in glob.glob(str(cache_dir) + "/*.json"):
        meta = _Meta.from_path(meta_path)
        if patterns and not any(fnmatch(meta.resource, p) for p in patterns):
            continue
        if meta.extraction_dir:
            cache_entries[meta.resource].extraction_dirs.append(meta)
        else:
            cache_entries[meta.resource].regular_files.append(meta)
        total_size += meta.size

    # Sort entries for each resource by creation time, newest first.
    for entry in cache_entries.values():
        entry.regular_files.sort(key=lambda meta: meta.creation_time, reverse=True)
        entry.extraction_dirs.sort(key=lambda meta: meta.creation_time, reverse=True)

    return total_size, cache_entries


def remove_cache_entries(patterns: List[str], cache_dir: Union[str, Path] = None) -> int:
    """
    Remove cache entries matching the given patterns.

    Returns the total reclaimed space in bytes.
    """
    total_size, cache_entries = _find_entries(patterns=patterns, cache_dir=cache_dir)
    for resource, entry in cache_entries.items():
        for meta in entry.regular_files:
            logger.info("Removing cached version of %s at %s", resource, meta.cached_path)
            os.remove(meta.cached_path)
            if os.path.exists(meta.cached_path + ".lock"):
                os.remove(meta.cached_path + ".lock")
            os.remove(meta.cached_path + ".json")
        for meta in entry.extraction_dirs:
            logger.info("Removing extracted version of %s at %s", resource, meta.cached_path)
            shutil.rmtree(meta.cached_path)
            if os.path.exists(meta.cached_path + ".lock"):
                os.remove(meta.cached_path + ".lock")
            os.remove(meta.cached_path + ".json")
    return total_size


def inspect_cache(patterns: List[str] = None, cache_dir: Union[str, Path] = None):
    """
    Print out useful information about the cache directory.
    """
    from allennlp.common.util import format_timedelta, format_size

    cache_dir = os.path.expanduser(cache_dir or CACHE_DIRECTORY)

    # Gather cache entries by resource.
    total_size, cache_entries = _find_entries(patterns=patterns, cache_dir=cache_dir)

    if patterns:
        print(f"Cached resources matching {patterns}:")
    else:
        print("Cached resources:")

    for resource, entry in sorted(
        cache_entries.items(),
        # Sort by creation time, latest first.
        key=lambda x: max(
            0 if not x[1][0] else x[1][0][0].creation_time,
            0 if not x[1][1] else x[1][1][0].creation_time,
        ),
        reverse=True,
    ):
        print("\n-", resource)
        if entry.regular_files:
            td = timedelta(seconds=time.time() - entry.regular_files[0].creation_time)
            n_versions = len(entry.regular_files)
            size = entry.regular_files[0].size
            print(
                f"  {n_versions} {'versions' if n_versions > 1 else 'version'} cached, "
                f"latest {format_size(size)} from {format_timedelta(td)} ago"
            )
        if entry.extraction_dirs:
            td = timedelta(seconds=time.time() - entry.extraction_dirs[0].creation_time)
            n_versions = len(entry.extraction_dirs)
            size = entry.extraction_dirs[0].size
            print(
                f"  {n_versions} {'versions' if n_versions > 1 else 'version'} extracted, "
                f"latest {format_size(size)} from {format_timedelta(td)} ago"
            )
    print(f"\nTotal size: {format_size(total_size)}")
