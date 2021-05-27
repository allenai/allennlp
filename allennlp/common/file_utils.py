"""
Utilities for working with the local dataset cache.
"""

from contextlib import contextmanager
import glob
import io
import os
import logging
import tempfile
import json
from abc import ABC
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
    MutableMapping,
)
from hashlib import sha256
from functools import wraps
from weakref import WeakValueDictionary
from zipfile import ZipFile, is_zipfile
import tarfile
import shutil
import pickle
import time
import warnings

import boto3
import botocore
import torch
from filelock import FileLock as _FileLock
from google.cloud import storage
from google.api_core.exceptions import NotFound
import numpy as np
from overrides import overrides
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import lmdb
from torch import Tensor
import huggingface_hub as hf_hub

from allennlp.version import VERSION
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


class FileLock(_FileLock):
    """
    This is just a subclass of the `FileLock` class from the `filelock` library, except that
    it adds an additional argument to the `__init__` method: `read_only_ok`.

    By default this flag is `False`, which an exception will be thrown when a lock
    can't be acquired due to lack of write permissions.
    But if this flag is set to `True`, a warning will be emitted instead of an error when
    the lock already exists but the lock can't be acquired because write access is blocked.
    """

    def __init__(
        self, lock_file: Union[str, PathLike], timeout=-1, read_only_ok: bool = False
    ) -> None:
        super().__init__(str(lock_file), timeout=timeout)
        self._read_only_ok = read_only_ok

    @overrides
    def acquire(self, timeout=None, poll_interval=0.05):
        try:
            super().acquire(timeout=timeout, poll_intervall=poll_interval)
        except OSError as err:
            # OSError could be a lot of different things, but what we're looking
            # for in particular are permission errors, such as:
            #  - errno 1  - EPERM  - "Operation not permitted"
            #  - errno 13 - EACCES - "Permission denied"
            #  - errno 30 - EROFS  - "Read-only file system"
            if err.errno not in (1, 13, 30):
                raise

            if os.path.isfile(self._lock_file) and self._read_only_ok:
                warnings.warn(
                    f"Lacking permissions required to obtain lock '{self._lock_file}'. "
                    "Race conditions are possible if other processes are writing to the same resource.",
                    UserWarning,
                )
            else:
                raise


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


def check_tarfile(tar_file: tarfile.TarFile):
    """Tar files can contain files outside of the extraction directory, or symlinks that point
    outside the extraction directory. We also don't want any block devices fifos, or other
    weird file types extracted. This checks for those issues and throws an exception if there
    is a problem."""
    base_path = os.path.join("tmp", "pathtest")
    base_path = os.path.normpath(base_path)

    def normalize_path(path: str) -> str:
        path = path.rstrip("/")
        path = path.replace("/", os.sep)
        path = os.path.join(base_path, path)
        path = os.path.normpath(path)
        return path

    for tarinfo in tar_file:
        if not (
            tarinfo.isreg()
            or tarinfo.isdir()
            or tarinfo.isfile()
            or tarinfo.islnk()
            or tarinfo.issym()
        ):
            raise ValueError(
                f"Tar file {str(tar_file.name)} contains invalid member {tarinfo.name}."
            )

        target_path = normalize_path(tarinfo.name)
        if os.path.commonprefix([base_path, target_path]) != base_path:
            raise ValueError(
                f"Tar file {str(tar_file.name)} is trying to create a file outside of its extraction directory."
            )

        if tarinfo.islnk() or tarinfo.issym():
            target_path = normalize_path(tarinfo.linkname)
            if os.path.commonprefix([base_path, target_path]) != base_path:
                raise ValueError(
                    f"Tar file {str(tar_file.name)} is trying to link to a file "
                    "outside of its extraction directory."
                )


def cached_path(
    url_or_filename: Union[str, PathLike],
    cache_dir: Union[str, Path] = None,
    extract_archive: bool = False,
    force_extract: bool = False,
) -> str:
    """
    Given something that might be a URL or local path, determine which.
    If it's a remote resource, download the file and cache it, and
    then return the path to the cached file. If it's already a local path,
    make sure the file exists and return the path.

    For URLs, "http://", "https://", "s3://", "gs://", and "hf://" are all supported.
    The latter corresponds to the HuggingFace Hub.

    For example, to download the PyTorch weights for the model `epwalsh/bert-xsmall-dummy`
    on HuggingFace, you could do:

    ```python
    cached_path("hf://epwalsh/bert-xsmall-dummy/pytorch_model.bin")
    ```

    For paths or URLs that point to a tarfile or zipfile, you can also add a path
    to a specific file to the `url_or_filename` preceeded by a "!", and the archive will
    be automatically extracted (provided you set `extract_archive` to `True`),
    returning the local path to the specific file. For example:

    ```python
    cached_path("model.tar.gz!weights.th", extract_archive=True)
    ```

    # Parameters

    url_or_filename : `Union[str, Path]`
        A URL or path to parse and possibly download.

    cache_dir : `Union[str, Path]`, optional (default = `None`)
        The directory to cache downloads.

    extract_archive : `bool`, optional (default = `False`)
        If `True`, then zip or tar.gz archives will be automatically extracted.
        In which case the directory is returned.

    force_extract : `bool`, optional (default = `False`)
        If `True` and the file is an archive file, it will be extracted regardless
        of whether or not the extracted directory already exists.

        !!! Warning
            Use this flag with caution! This can lead to race conditions if used
            from multiple processes on the same file.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if not isinstance(url_or_filename, str):
        url_or_filename = str(url_or_filename)

    file_path: str
    extraction_path: Optional[str] = None

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

    if parsed.scheme in ("http", "https", "s3", "hf", "gs"):
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

            if (
                extract_archive
                and os.path.isfile(file_path)
                and (is_zipfile(file_path) or tarfile.is_tarfile(file_path))
            ):
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
        # need to create a lock file and extract again unless `force_extract=True`.
        if os.path.isdir(extraction_path) and os.listdir(extraction_path) and not force_extract:
            return extraction_path

        # Extract it.
        with FileLock(extraction_path + ".lock"):
            # Check again if the directory exists now that we've acquired the lock.
            if os.path.isdir(extraction_path) and os.listdir(extraction_path):
                if force_extract:
                    logger.warning(
                        "Extraction directory for %s (%s) already exists, "
                        "overwriting it since 'force_extract' is 'True'",
                        url_or_filename,
                        extraction_path,
                    )
                else:
                    return extraction_path

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
                    check_tarfile(tar_file)
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
    return parsed.scheme in ("http", "https", "s3", "gs") or os.path.exists(url_or_filename)


def _split_s3_path(url: str) -> Tuple[str, str]:
    return _split_cloud_path(url, "s3")


def _split_gcs_path(url: str) -> Tuple[str, str]:
    return _split_cloud_path(url, "gs")


def _split_cloud_path(url: str, provider: str) -> Tuple[str, str]:
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad {} path {}".format(provider, url))
    bucket_name = parsed.netloc
    provider_path = parsed.path
    # Remove '/' at beginning of path.
    if provider_path.startswith("/"):
        provider_path = provider_path[1:]
    return bucket_name, provider_path


def _s3_request(func: Callable):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except botocore.exceptions.ClientError as exc:
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


def _gcs_request(func: Callable):
    """
    Wrapper function for gcs requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except NotFound:
            raise FileNotFoundError("file {} not found".format(url))

    return wrapper


def _get_gcs_client():
    storage_client = storage.Client()
    return storage_client


def _get_gcs_blob(url: str) -> storage.blob.Blob:
    gcs_resource = _get_gcs_client()
    bucket_name, gcs_path = _split_gcs_path(url)
    bucket = gcs_resource.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    return blob


@_gcs_request
def _gcs_md5(url: str) -> Optional[str]:
    """Get GCS object's md5."""
    blob = _get_gcs_blob(url)
    return blob.md5_hash


@_gcs_request
def _gcs_get(url: str, temp_filename: str) -> None:
    """Pull a file directly from GCS."""
    blob = _get_gcs_blob(url)
    blob.download_to_filename(temp_filename)


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
        raise OSError(
            "HEAD request failed for url {} with status code {}".format(url, response.status_code)
        )
    return response.headers.get("ETag")


def _http_get(url: str, temp_file: IO) -> None:
    with _session_with_backoff() as session:
        req = session.get(url, stream=True)
        req.raise_for_status()
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


def _serialize(data):
    buffer = pickle.dumps(data, protocol=-1)
    return np.frombuffer(buffer, dtype=np.uint8)


class TensorCache(MutableMapping[str, Tensor], ABC):
    """
    This is a key-value store, mapping strings to tensors. The data is kept on disk,
    making this class useful as a cache for storing tensors.

    `TensorCache` is also safe to access from multiple processes at the same time, so
    you can use it in distributed training situations, or from multiple training
    runs at the same time.
    """

    def __init__(
        self,
        filename: Union[str, PathLike],
        *,
        map_size: int = 1024 * 1024 * 1024 * 1024,
        read_only: bool = False,
    ) -> None:
        """
        Creates a `TensorCache` by either opening an existing one on disk, or creating
        a new one. Its interface is almost exactly like a Python dictionary, where the
        keys are strings and the values are `torch.Tensor`.

        Parameters
        ----------
        filename: `str`
            Path to the location of the cache
        map_size: `int`, optional, defaults to 1TB
            This is the maximum size the cache will ever grow to. On reasonable operating
            systems, there is no penalty to making this a large value.
            `TensorCache` uses a memory-mapped file to store the data. When the file is
            first opened, we have to give the maximum size it can ever grow to. This is
            that number. Reasonable operating systems don't actually allocate that space
            until it is really needed.
        """
        filename = str(filename)

        cpu_count = os.cpu_count() or 1
        if os.path.exists(filename):
            if os.path.isfile(filename):
                # If the file is not writable, set read_only to True, but issue a warning.
                if not os.access(filename, os.W_OK):
                    if not read_only:
                        warnings.warn(
                            f"File '{filename}' is read-only, so cache will be read-only",
                            UserWarning,
                        )
                    read_only = True
            else:
                # If it's not a file, raise an error.
                raise ValueError("Expect a file, found a directory instead")

        use_lock = True
        if read_only:
            # Check if the lock file is writable. If it's not, then we won't be able to use the lock.

            # This is always how lmdb names the lock file.
            lock_filename = filename + "-lock"
            if os.path.isfile(lock_filename):
                use_lock = os.access(lock_filename, os.W_OK)
            else:
                # If the lock file doesn't exist yet, then the directory needs to be writable in
                # order to create and use the lock file.
                use_lock = os.access(os.path.dirname(lock_filename), os.W_OK)

        if not use_lock:
            warnings.warn(
                f"Lacking permissions to use lock file on cache '{filename}'.\nUse at your own risk!",
                UserWarning,
            )

        self.lmdb_env = lmdb.open(
            str(filename),
            subdir=False,
            map_size=map_size,
            max_readers=cpu_count * 2,
            max_spare_txns=cpu_count * 2,
            metasync=False,
            sync=True,
            readahead=False,
            meminit=False,
            readonly=read_only,
            lock=use_lock,
        )

        # We have another cache here that makes sure we return the same object for the same key. Without it,
        # you would get a different tensor, using different memory, every time you call __getitem__(), even
        # if you call it with the same key.
        # The downside is that we can't keep self.cache_cache up to date when multiple processes modify the
        # cache at the same time. We can guarantee though that it is up to date as long as processes either
        # write new values, or read existing ones.
        self.cache_cache: MutableMapping[str, Tensor] = WeakValueDictionary()

    @property
    def read_only(self) -> bool:
        return self.lmdb_env.flags()["readonly"]

    def __contains__(self, key: object):
        if not isinstance(key, str):
            return False
        if key in self.cache_cache:
            return True
        encoded_key = key.encode()
        with self.lmdb_env.begin(write=False) as txn:
            result = txn.get(encoded_key)
            return result is not None

    def __getitem__(self, key: str):
        try:
            return self.cache_cache[key]
        except KeyError:
            encoded_key = key.encode()
            with self.lmdb_env.begin(write=False) as txn:
                buffer = txn.get(encoded_key)
                if buffer is None:
                    raise KeyError()
                tensor = torch.load(io.BytesIO(buffer), map_location="cpu")
            self.cache_cache[key] = tensor
            return tensor

    def __setitem__(self, key: str, tensor: torch.Tensor):
        if self.read_only:
            raise ValueError("cannot write to a read-only cache")

        tensor = tensor.cpu()
        encoded_key = key.encode()
        buffer = io.BytesIO()
        if tensor.storage().size() != np.prod(tensor.size()):
            tensor = tensor.clone()
        assert tensor.storage().size() == np.prod(tensor.size())
        torch.save(tensor.detach(), buffer, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        with self.lmdb_env.begin(write=True) as txn:
            txn.put(encoded_key, buffer.getbuffer())

        self.cache_cache[key] = tensor

    def __delitem__(self, key: str):
        if self.read_only:
            raise ValueError("cannot write to a read-only cache")

        encoded_key = key.encode()
        with self.lmdb_env.begin(write=True) as txn:
            txn.delete(encoded_key)

        try:
            del self.cache_cache[key]
        except KeyError:
            pass

    def __del__(self):
        if self.lmdb_env is not None:
            self.lmdb_env.close()
            self.lmdb_env = None

    def __len__(self):
        return self.lmdb_env.stat()["entries"]

    def __iter__(self):
        # It is not hard to implement this, but we have not needed it so far.
        raise NotImplementedError()


class CacheFile:
    """
    This is a context manager that makes robust caching easier.

    On `__enter__`, an IO handle to a temporarily file is returned, which can
    be treated as if it's the actual cache file.

    On `__exit__`, the temporarily file is renamed to the cache file. If anything
    goes wrong while writing to the temporary file, it will be removed.
    """

    def __init__(
        self, cache_filename: Union[PathLike, str], mode: str = "w+b", suffix: str = ".tmp"
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


class LocalCacheResource:
    """
    This is a context manager that can be used to fetch and cache arbitrary resources locally
    using the same mechanisms that `cached_path` uses for remote resources.

    It can be used, for example, when you want to cache the result of an expensive computation.

    # Examples

    ```python
    with LocalCacheResource("long-computation", "v1") as cache:
        if cache.cached():
            with cache.reader() as f:
                # read from cache
        else:
            with cache.writer() as f:
                # do the computation
                # ...
                # write to cache
    ```
    """

    def __init__(self, resource_name: str, version: str, cache_dir: str = CACHE_DIRECTORY) -> None:
        self.resource_name = resource_name
        self.version = version
        self.cache_dir = cache_dir
        self.path = os.path.join(self.cache_dir, _resource_to_filename(resource_name, version))
        self.file_lock = FileLock(self.path + ".lock")

    def cached(self) -> bool:
        return os.path.exists(self.path)

    @contextmanager
    def writer(self, mode="w"):
        if self.cached():
            raise ValueError(
                f"local cache of {self.resource_name} (version '{self.version}') already exists!"
            )

        with CacheFile(self.path, mode=mode) as f:
            yield f

        meta = _Meta(
            resource=self.resource_name,
            cached_path=self.path,
            creation_time=time.time(),
            etag=self.version,
            size=_get_resource_size(self.path),
        )
        meta.to_file()

    @contextmanager
    def reader(self, mode="r"):
        if not self.cached():
            raise ValueError(
                f"local cache of {self.resource_name} (version '{self.version}') does not exist yet!"
            )

        with open(self.path, mode) as f:
            yield f

    def __enter__(self):
        self.file_lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file_lock.release()
        if exc_value is None:
            return True
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


def _hf_hub_download(
    url, model_identifier: str, filename: Optional[str], cache_dir: Union[str, Path]
) -> str:
    revision: Optional[str]
    if "@" in model_identifier:
        repo_id = model_identifier.split("@")[0]
        revision = model_identifier.split("@")[1]
    else:
        repo_id = model_identifier
        revision = None

    if filename is not None:
        hub_url = hf_hub.hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
        cache_path = str(
            hf_hub.cached_download(
                url=hub_url,
                library_name="allennlp",
                library_version=VERSION,
                cache_dir=cache_dir,
            )
        )
        # HF writes it's own meta '.json' file which uses the same format we used to use and still
        # support, but is missing some fields that we like to have.
        # So we overwrite it when it we can.
        with FileLock(cache_path + ".lock", read_only_ok=True):
            meta = _Meta.from_path(cache_path + ".json")
            # The file HF writes will have 'resource' set to the 'http' URL corresponding to the 'hf://' URL,
            # but we want 'resource' to be the original 'hf://' URL.
            if meta.resource != url:
                meta.resource = url
                meta.to_file()
    else:
        cache_path = str(hf_hub.snapshot_download(repo_id, revision=revision, cache_dir=cache_dir))
        # Need to write the meta file for snapshot downloads if it doesn't exist.
        with FileLock(cache_path + ".lock", read_only_ok=True):
            if not os.path.exists(cache_path + ".json"):
                meta = _Meta(
                    resource=url,
                    cached_path=cache_path,
                    creation_time=time.time(),
                    extraction_dir=True,
                    size=_get_resource_size(cache_path),
                )
                meta.to_file()
    return cache_path


# TODO(joelgrus): do we want to do checksums or anything like that?
def get_from_cache(url: str, cache_dir: Union[str, Path] = None) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    if url.startswith("hf://"):
        # Remove the 'hf://' prefix
        identifier = url[5:]

        if identifier.count("/") > 1:
            filename = "/".join(identifier.split("/")[2:])
            model_identifier = "/".join(identifier.split("/")[:2])
            return _hf_hub_download(url, model_identifier, filename, cache_dir)
        elif identifier.count("/") == 1:
            # 'hf://' URLs like 'hf://xxxx/yyyy' are potentially ambiguous,
            # because this could refer to either:
            #  1. the file 'yyyy' in the 'xxxx' repository, or
            #  2. the repo 'yyyy' under the user/org name 'xxxx'.
            # We default to (1), but if we get a 404 error then we try (2).
            try:
                model_identifier, filename = identifier.split("/")
                return _hf_hub_download(url, model_identifier, filename, cache_dir)
            except requests.exceptions.HTTPError as exc:
                if exc.response.status_code == 404:
                    return _hf_hub_download(url, identifier, None, cache_dir)
                raise
        else:
            return _hf_hub_download(url, identifier, None, cache_dir)

    # Get eTag to add to filename, if it exists.
    try:
        if url.startswith("s3://"):
            etag = _s3_etag(url)
        elif url.startswith("gs://"):
            etag = _gcs_md5(url)
        else:
            etag = _http_etag(url)
    except (requests.exceptions.ConnectionError, botocore.exceptions.EndpointConnectionError):
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
    with FileLock(cache_path + ".lock", read_only_ok=True):
        if os.path.exists(cache_path):
            logger.info("cache of %s is up-to-date", url)
        else:
            with CacheFile(cache_path) as cache_file:
                logger.info("%s not found in cache, downloading to %s", url, cache_path)

                # GET file object
                if url.startswith("s3://"):
                    _s3_get(url, cache_file)
                elif url.startswith("gs://"):
                    _gcs_get(url, cache_file.name)
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
    filename: Union[str, PathLike], mode: str = "rt", encoding: Optional[str] = "UTF-8", **kwargs
):
    if not isinstance(filename, str):
        filename = str(filename)
    open_fn: Callable = open

    if filename.endswith(".gz"):
        import gzip

        open_fn = gzip.open
    elif filename.endswith(".bz2"):
        import bz2

        open_fn = bz2.open
    return open_fn(cached_path(filename), mode=mode, encoding=encoding, **kwargs)


def text_lines_from_file(filename: Union[str, PathLike], strip_lines: bool = True) -> Iterator[str]:
    with open_compressed(filename, "rt", encoding="UTF-8", errors="replace") as p:
        if strip_lines:
            for line in p:
                yield line.strip()
        else:
            yield from p


def json_lines_from_file(filename: Union[str, PathLike]) -> Iterable[Union[list, dict]]:
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
