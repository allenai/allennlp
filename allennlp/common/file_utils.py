"""
Utilities for working with the local dataset cache.
"""
import io
import gzip
import zipfile
from typing import Tuple
import os
import base64
import logging
import shutil
import tempfile
from urllib.parse import urlparse

import requests

from allennlp.common.tqdm import Tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CACHE_ROOT = os.getenv('ALLENNLP_CACHE_ROOT', os.path.expanduser(os.path.join('~', '.allennlp')))
DATASET_CACHE = os.path.join(CACHE_ROOT, "datasets")


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
        _, temp_filename = tempfile.mkstemp()
        logger.info("%s not found in cache, downloading to %s", url, temp_filename)

        # GET file object
        req = requests.get(url, stream=True)
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = Tqdm.tqdm(unit="B", total=total)
        with open(temp_filename, 'wb') as temp_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)

        progress.close()

        logger.info("copying %s to cache at %s", temp_filename, cache_path)
        shutil.copyfile(temp_filename, cache_path)
        logger.info("removing temp file %s", temp_filename)
        os.remove(temp_filename)

    return cache_path


def get_file_extension(path: str, dot=True, lower: bool = True):
    """ Returns the file extension, by default case-lowered and including the dot. """
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


class CompressedFileUtils:

    SUPPORTED_FORMATS = {'.gz', '.zip'}
    READ_MODE_CHOICES = {'t', 'rt', 'b', 'rb'}
    DEFAULT_ENCODING = 'utf-8'

    @staticmethod
    def read(path: str, mode: str = 't', encoding: str = None,
             file_format: str = None):
        """
        Open an eventually compressed file in binary or text mode (default: text mode
        with utf-8 encoding). The currently supported compressed file formats are: gzip, zip.

        Arguments:
        mode: str
            Aliases for text mode: 't' and 'rt'; aliases for binary mode: 'b' and 'rb'.
        file_format: str
            If ``file_format == None``, the format is inferred from the extension.
            If the file format (specified or inferred) is not supported, an exception is raised.

        Returns:
        When used in text mode, it returns a :class:`io.TextIOWrapper`.
        When used in binary mode, the return type depends on the specific input file format.
        """
        read_mode_choices = CompressedFileUtils.READ_MODE_CHOICES
        if mode not in read_mode_choices:
            raise ValueError("Invalid mode: {}. Supported modes are: {}"
                             .format(mode, read_mode_choices))

        file_format = file_format or get_file_extension(path)

        if file_format == ".gz":
            logger.info("Opening gzipped file: %s", path)
            binary_reader = gzip.open(path, "rb")

        elif file_format == ".zip":
            logger.info("Opening zipped file: %s", path)
            zfile = zipfile.ZipFile(path, "r")
            assert len(zfile.namelist()) == 1, "Multiple files are contained in the zip archive " + path
            filename = zfile.namelist()[0]
            binary_reader = zfile.open(filename, "r")

        else:
            raise ValueError('Unsupported file format: {}. Supported formats are: {}'
                             .format(file_format, CompressedFileUtils.SUPPORTED_FORMATS))

        if mode == 't':
            encoding = encoding or CompressedFileUtils.DEFAULT_ENCODING
            return io.TextIOWrapper(binary_reader, encoding=encoding)
        else:
            if encoding is not None:    # same behavior of built-in open()
                raise ValueError("Binary mode doesn't take an encoding argument")
            return binary_reader


def read_maybe_compressed_file(path: str, mode: str = 't', encoding: str = None,
                               file_format: str = None):
    """
    If the file format is in :const:`CompressedFileUtils.SUPPORTED_FORMATS`, the file is
    opened using :func:`CompressedFileUtils.read`, otherwise it's assumed to be
    uncompressed and it's opened using the built-in :func:`open` function.

    The default encoding for text files is :const:`CompressedFileUtils.DEFAULT_ENCODING`.
    """
    file_format = file_format or get_file_extension(path)

    if file_format in CompressedFileUtils.SUPPORTED_FORMATS:
        return CompressedFileUtils.read(path, mode, encoding, file_format)
    else:
        assert mode in CompressedFileUtils.READ_MODE_CHOICES
        logger.info("Reading the file assuming it's not compressed: %s", path)
        if mode == 't': mode = 'rt'
        if mode == 'b': mode = 'rb'
        return open(path, mode, encoding=encoding)
