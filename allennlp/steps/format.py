import bz2
import gzip
import lzma
import mmap
import pathlib
from collections import abc
from os import PathLike
from typing import TypeVar, Generic, Union, Optional

import dill
import xxhash

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError

T = TypeVar("T")


class Format(Registrable, Generic[T]):
    """Formats write objects to directories and read them back out."""

    VERSION: int = NotImplemented
    default_implementation = "dill"

    def write(self, artifact: T, dir: Union[str, PathLike]):
        raise NotImplementedError()

    def read(self, dir: Union[str, PathLike]) -> T:
        raise NotImplementedError()

    def checksum(self, dir: Union[str, PathLike]) -> str:
        """The default checksum mechanism computes a checksum of all the files in the
        directory except for `metadata.json`."""
        dir = pathlib.Path(dir)
        files = []
        for file in dir.rglob("*"):
            if file.name == "metadata.json":
                continue
            if not (file.is_file() or file.is_symlink()):
                continue
            files.append(file)
        files.sort()

        h = xxhash.xxh128()
        for file in files:
            with file.open("rb") as f:
                with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
                    h.update(m)
        return h.hexdigest()


@Format.register("dill")
class DillFormat(Format):
    """This format writes the artifact as a single file using dill (a drop-in replacement for pickle).
    Optionally, it can compress the data."""

    VERSION = 1

    OPEN_FUNCTIONS = {
        None: open,
        "None": open,
        "none": open,
        "null": open,
        "gz": gzip.open,
        "gzip": gzip.open,
        "bz": bz2.open,
        "bz2": bz2.open,
        "bzip": bz2.open,
        "bzip2": bz2.open,
        "lzma": lzma.open,
    }

    SUFFIXES = {
        open: "",
        gzip.open: ".gz",
        bz2.open: ".bz2",
        lzma.open: ".xz",
    }

    def __init__(self, compress: Optional[str] = None):
        try:
            self.open = self.OPEN_FUNCTIONS[compress]
        except KeyError:
            raise ConfigurationError(f"The {compress} compression format does not exist.")

    def write(self, artifact: T, dir: Union[str, PathLike]):
        filename = pathlib.Path(dir) / ("data.dill" + self.SUFFIXES[self.open])
        with self.open(filename, "wb") as f:
            pickler = dill.Pickler(file=f)
            pickler.dump(self.VERSION)
            if hasattr(artifact, "__next__"):
                pickler.dump(True)
                for item in artifact:
                    pickler.dump(item)
            else:
                pickler.dump(False)
                pickler.dump(artifact)

    def read(self, dir: Union[str, PathLike]) -> T:
        filename = pathlib.Path(dir) / ("data.dill" + self.SUFFIXES[self.open])
        with self.open(filename, "rb") as f:
            unpickler = dill.Unpickler(file=f)
            version = unpickler.load()
            if version > self.VERSION:
                raise ValueError(
                    f"File {filename} is too recent for this version of {self.__class__}."
                )
            iterator = unpickler.load()
            if iterator:
                return DillFormatIterator(filename)
            else:
                return unpickler.load()


class DillFormatIterator(abc.Iterator):
    """This class is used so we can return an iterator from `DillFormat.read()`."""

    def __init__(self, filename: Union[str, PathLike]):
        if filename.endswith(".gz"):
            open_fn = gzip.open
        elif filename.endswith(".bz2"):
            open_fn = bz2.open
        elif filename.endswith(".xz"):
            open_fn = lzma.open
        else:
            open_fn = open
        self.f = open_fn(filename)
        self.unpickler = dill.Unpickler(self.f)
        version = self.unpickler.load()
        if version > DillFormat.VERSION:
            raise ValueError(f"File {filename} is too recent for this version of {self.__class__}.")
        iterator = self.unpickler.load()
        if not iterator:
            raise ValueError(
                f"Tried to open {filename} as an iterator, but it does not store an iterator."
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self.f is None:
            raise StopIteration()
        try:
            return self.unpickler.load()
        except EOFError:
            self.f.close()
            self.f = None
            raise StopIteration()
