import bz2
import gzip
import lzma
import mmap
import pathlib
from abc import abstractmethod
from os import PathLike
from typing import (
    TypeVar,
    Generic,
    Union,
    Optional,
    Callable,
    Dict,
    Iterable,
    cast,
    Iterator,
)

import dill
import torch
import xxhash
from typing.io import IO

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError

T = TypeVar("T")


class Format(Registrable, Generic[T]):
    """
    Formats write objects to directories and read them back out.

    In the context of AllenNLP, the objects that are written by formats are usually
    results from `Step`s.
    """

    VERSION: int = NotImplemented
    """
    Formats can have versions. Versions are part of a step's unique signature, part of `Step.unique_id()`,
    so when a step's format changes, that will cause the step to be recomputed.
    """

    default_implementation = "dill"

    @abstractmethod
    def write(self, artifact: T, dir: Union[str, PathLike]):
        """Writes the `artifact` to the directory at `dir`."""
        raise NotImplementedError()

    @abstractmethod
    def read(self, dir: Union[str, PathLike]) -> T:
        """Reads an artifact from the directory at `dir` and returns it."""
        raise NotImplementedError()

    def checksum(self, dir: Union[str, PathLike]) -> str:
        """
        Produces a checksum of a serialized artifact.

        The default checksum mechanism computes a checksum of all the files in the
        directory except for `metadata.json`.
        """
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
class DillFormat(Format[T], Generic[T]):
    """This format writes the artifact as a single file using dill (a drop-in replacement for pickle).
    Optionally, it can compress the data. This is very flexible, but not always the fastest.

    This format has special support for iterables. If you write an iterator, it will consume the
    iterator. If you read an iterator, it will read the iterator lazily.
    """

    VERSION = 1

    OPEN_FUNCTIONS: Dict[Optional[str], Callable[[PathLike, str], IO]] = {
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
                for item in cast(Iterable, artifact):
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
                return DillFormatIterator(filename)  # type: ignore
            else:
                return unpickler.load()


class DillFormatIterator(Iterator[T], Generic[T]):
    """This class is used so we can return an iterator from `DillFormat.read()`."""

    def __init__(self, filename: Union[str, PathLike]):
        filename = str(filename)
        open_fn: Callable
        if filename.endswith(".gz"):
            open_fn = gzip.open
        elif filename.endswith(".bz2"):
            open_fn = bz2.open
        elif filename.endswith(".xz"):
            open_fn = lzma.open
        else:
            open_fn = open
        self.f = open_fn(filename, "rb")
        self.unpickler = dill.Unpickler(self.f)
        version = self.unpickler.load()
        if version > DillFormat.VERSION:
            raise ValueError(f"File {filename} is too recent for this version of {self.__class__}.")
        iterator = self.unpickler.load()
        if not iterator:
            raise ValueError(
                f"Tried to open {filename} as an iterator, but it does not store an iterator."
            )

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self.f is None:
            raise StopIteration()
        try:
            return self.unpickler.load()
        except EOFError:
            self.f.close()
            self.f = None
            raise StopIteration()


@Format.register("torch")
class TorchFormat(Format[T], Generic[T]):
    """
    This format writes the artifact using torch.save().

    Unlike `DillFormat`, this has no special support for iterators.
    """

    VERSION = 2

    def write(self, artifact: T, dir: Union[str, PathLike]):
        filename = pathlib.Path(dir) / "data.pt"
        with open(filename, "wb") as f:
            torch.save((self.VERSION, artifact), f, pickle_module=dill)

    def read(self, dir: Union[str, PathLike]) -> T:
        filename = pathlib.Path(dir) / "data.pt"
        with open(filename, "rb") as f:
            version, artifact = torch.load(f, pickle_module=dill, map_location=torch.device("cpu"))
            if version > self.VERSION:
                raise ValueError(
                    f"File {filename} is too recent for this version of {self.__class__}."
                )
            return artifact
