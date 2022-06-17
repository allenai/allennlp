"""
Utilities for working with the local dataset cache.
"""
import bz2
import gzip
import lzma
import weakref
from contextlib import contextmanager
import glob
import io
import os
import logging
import json
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from fnmatch import fnmatch
from os import PathLike
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    Union,
    Callable,
    Set,
    List,
    Iterator,
    Iterable,
    Dict,
    NamedTuple,
    MutableMapping,
)
from weakref import WeakValueDictionary
import shutil
import pickle
import time
import warnings

import cached_path as _cached_path
from cached_path import (  # noqa: F401
    resource_to_filename as _resource_to_filename,
    check_tarfile,
    is_url_or_existing_file,
    find_latest_cached as _find_latest_cached,
)
from cached_path.cache_file import CacheFile
from cached_path.common import PathOrStr
from cached_path.file_lock import FileLock
from cached_path.meta import Meta as _Meta
import torch
import numpy as np
import lmdb
from torch import Tensor


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


def filename_to_url(filename: str, cache_dir: Union[str, Path] = None) -> Tuple[str, str]:
    """
    Return the url and etag (which may be `None`) stored for `filename`.
    Raise `FileNotFoundError` if `filename` or its stored metadata do not exist.
    """
    return _cached_path.filename_to_url(filename, cache_dir=cache_dir or CACHE_DIRECTORY)


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
    return str(
        _cached_path.cached_path(
            url_or_filename,
            cache_dir=cache_dir or CACHE_DIRECTORY,
            extract_archive=extract_archive,
            force_extract=force_extract,
        )
    )


def _serialize(data):
    buffer = pickle.dumps(data, protocol=-1)
    return np.frombuffer(buffer, dtype=np.uint8)


_active_tensor_caches: MutableMapping[int, "TensorCache"] = weakref.WeakValueDictionary()


def _unique_file_id(path: Union[str, PathLike]) -> int:
    result = os.stat(path).st_ino
    assert result != 0
    return result


class TensorCache(MutableMapping[str, Tensor], ABC):
    """
    This is a key-value store, mapping strings to tensors. The data is kept on disk,
    making this class useful as a cache for storing tensors.

    `TensorCache` is also safe to access from multiple processes at the same time, so
    you can use it in distributed training situations, or from multiple training
    runs at the same time.
    """

    def __new__(cls, filename: Union[str, PathLike], *, read_only: bool = False, **kwargs):
        # This mechanism makes sure we re-use open lmdb file handles. Lmdb has a problem when the same file is
        # opened by the same process multiple times. This is our workaround.
        filename = str(filename)
        try:
            result = _active_tensor_caches.get(_unique_file_id(filename))
        except FileNotFoundError:
            result = None
        if result is None:
            result = super(TensorCache, cls).__new__(cls)
        return result

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
        self.lmdb_env: lmdb.Environment
        if hasattr(self, "lmdb_env"):
            # We're being initialized again after a cache hit in _active_tensor_caches, thanks
            # to __new__. In this case, we may have to upgrade to read/write, but other than
            # that we are good to go.
            if read_only:
                return
            if not self.read_only:
                return

            # Upgrade a read-only lmdb env to a read/write lmdb env.
            filename = self.lmdb_env.path()
            old_info = self.lmdb_env.info()

            self.lmdb_env.close()
            self.lmdb_env = lmdb.open(
                filename,
                map_size=old_info["map_size"],
                subdir=False,
                metasync=False,
                sync=True,
                readahead=False,
                meminit=False,
                readonly=False,
                lock=True,
            )
        else:
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
                filename,
                subdir=False,
                map_size=map_size,
                max_readers=cpu_count * 4,
                max_spare_txns=cpu_count * 4,
                metasync=False,
                sync=True,
                readahead=False,
                meminit=False,
                readonly=read_only,
                lock=use_lock,
            )
            _active_tensor_caches[_unique_file_id(filename)] = self

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


_SUFFIXES: Dict[Callable, str] = {
    open: "",
    gzip.open: ".gz",
    bz2.open: ".bz2",
    lzma.open: ".xz",
}


def open_compressed(
    filename: Union[str, PathLike],
    mode: str = "rt",
    encoding: Optional[str] = "UTF-8",
    **kwargs,
):
    if not isinstance(filename, str):
        filename = str(filename)

    open_fn: Callable
    filename = str(filename)
    for open_fn, suffix in _SUFFIXES.items():
        if len(suffix) > 0 and filename.endswith(suffix):
            break
    else:
        open_fn = open

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


def hardlink_or_copy(source: PathOrStr, dest: PathOrStr):
    try:
        os.link(source, dest)
    except OSError as e:
        if e.errno in {18, 95}:  # Cross-device link and Windows
            shutil.copy(source, dest)
        else:
            raise
