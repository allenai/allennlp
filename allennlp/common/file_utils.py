"""
Utilities for working with the local dataset cache.
"""
import io
import gzip
import zipfile
from contextlib import contextmanager
from typing import Tuple, Optional, ContextManager
import os
from hashlib import sha256
import logging
import shutil
import tempfile
from urllib.parse import urlparse
import json

import requests

from allennlp.common.tqdm import Tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CACHE_ROOT = os.getenv('ALLENNLP_CACHE_ROOT', os.path.expanduser(os.path.join('~', '.allennlp')))
DATASET_CACHE = os.path.join(CACHE_ROOT, "datasets")


def url_to_filename(url: str, etag: str = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def filename_to_url(filename: str, cache_dir: str = None) -> Tuple[str, str]:
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``FileNotFoundError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = DATASET_CACHE

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError("file {} not found".format(cache_path))

    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError("file {} not found".format(meta_path))

    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']

    return url, etag


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

    os.makedirs(cache_dir, exist_ok=True)

    # make HEAD request to check ETag
    response = requests.head(url)
    if response.status_code != 200:
        raise IOError("HEAD request failed for url {}".format(url))

    # add ETag to filename if it exists
    etag = response.headers.get("ETag")
    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
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

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path


def get_file_extension(path: str, dot=True, lower: bool = True):
    """ Returns the file extension, by default case-lowered and including the dot. """
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


class CompressedFileUtils:

    SUPPORTED_FORMATS = {'.gz', '.zip'}
    MODE_CHOICES = {'r', 'rt', 'rb'}
    DEFAULT_ENCODING = 'utf-8'

    @staticmethod
    def open(path: str, mode: str = 'rt', encoding: Optional[str] = None,
             file_format: Optional[str] = None) -> ContextManager[io.IOBase]:
        """
        Open an eventually compressed file in binary or text mode (default: text mode
        with utf-8 encoding). The currently supported compressed file formats are: gzip, zip.

        Parameters
        ----------
        path: str
            Path to the file to read
        mode: str
            Reading mode; it can be 'r' or 'rt' for text mode, 'rb' for binary mode.
        encoding: str
            Text encoding of the text file. This must be left to None when reading in binary mode
            or an exception will be raised.
        file_format: str
            If ``file_format == None``, the format is inferred from the extension.
            If the file format (specified or inferred) is not supported, an exception is raised.

        Returns
        -------
        file_handle: ContextManager[io.IOBase]
            a context manager (to be used with ``with``) that returns a file handle. The specific type
            of a file handle is a ``io.TextIOWrapper`` when reading in text mode. When reading in binary
            mode, the specific type depends on the file format.
        """
        if mode not in CompressedFileUtils.MODE_CHOICES:
            raise ValueError("Invalid mode: {}. Supported modes are: {}"
                             .format(mode, CompressedFileUtils.MODE_CHOICES))

        file_format = file_format or get_file_extension(path)

        if file_format == ".gz":
            logger.info("Opening gzipped file: %s", path)
            return gzip.open(path, mode, encoding=encoding)

        elif file_format == ".zip":
            logger.info("Opening zipped file: %s", path)
            return CompressedFileUtils._open_zipped_file(path, mode, encoding)
        else:
            raise ValueError('Unsupported file format: {}. Supported formats are: {}'
                             .format(file_format, CompressedFileUtils.SUPPORTED_FORMATS))

    @staticmethod
    @contextmanager
    def _open_zipped_file(path: str, mode: str = 'rt',
                          encoding: Optional[str] = None) -> ContextManager[io.IOBase]:
        zip_archive = zipfile.ZipFile(path, "r")
        namelist = zip_archive.namelist()
        if len(namelist) > 1:
            raise ValueError("Multiple files are contained in the zip archive " + path)
        binary_reader = zip_archive.open(namelist[0], "r")

        if mode == 'rb':
            # Binary mode
            if encoding is not None:  # same behavior of built-in open()
                raise ValueError("Binary mode doesn't take an encoding argument")
            content_file = binary_reader
        else:
            # Text mode
            encoding = encoding or CompressedFileUtils.DEFAULT_ENCODING
            content_file = io.TextIOWrapper(binary_reader, encoding=encoding)

        yield content_file

        content_file.close()
        zip_archive.close()


def open_maybe_compressed_file(path: str, mode: str = 'rt', encoding: str = None,
                               file_format: str = None) -> ContextManager[io.IOBase]:
    """
    If the file format is in :const:`CompressedFileUtils.SUPPORTED_FORMATS`, the file is
    opened using :func:`CompressedFileUtils.read`, otherwise it's assumed to be
    uncompressed and it's opened using the built-in :func:`open` function.

    The default encoding for text files is :const:`CompressedFileUtils.DEFAULT_ENCODING`.
    """
    file_format = file_format or get_file_extension(path)

    if file_format in CompressedFileUtils.SUPPORTED_FORMATS:
        return CompressedFileUtils.open(path, mode, encoding, file_format)
    else:
        if mode not in CompressedFileUtils.MODE_CHOICES:
            raise ValueError("Invalid mode: {}. Supported modes are: {}"
                             .format(mode, CompressedFileUtils.MODE_CHOICES))
        logger.info("Reading the file assuming it's not compressed: %s", path)
        return open(path, mode, encoding=encoding)
