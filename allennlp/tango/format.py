"""
*AllenNLP Tango is an experimental API and parts of it might change or disappear
every time we release a new version.*
"""

import bz2
import dataclasses
import gzip
import importlib
import json
import logging
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
    Any,
    List,
)

import dill
import torch
import xxhash
from typing.io import IO

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.logging import AllenNlpLogger

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


_OPEN_FUNCTIONS: Dict[Optional[str], Callable[[PathLike, str], IO]] = {
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

_SUFFIXES: Dict[Callable, str] = {
    open: "",
    gzip.open: ".gz",
    bz2.open: ".bz2",
    lzma.open: ".xz",
}


def _open_compressed(filename: Union[str, PathLike], mode: str) -> IO:
    open_fn: Callable
    filename = str(filename)
    for open_fn, suffix in _SUFFIXES.items():
        if len(suffix) > 0 and filename.endswith(suffix):
            break
    else:
        open_fn = open
    return open_fn(filename, mode)


@Format.register("dill")
class DillFormat(Format[T], Generic[T]):
    """This format writes the artifact as a single file using dill (a drop-in replacement for pickle).
    Optionally, it can compress the data. This is very flexible, but not always the fastest.

    This format has special support for iterables. If you write an iterator, it will consume the
    iterator. If you read an iterator, it will read the iterator lazily.
    """

    VERSION = 1

    def __init__(self, compress: Optional[str] = None):
        try:
            self.open = _OPEN_FUNCTIONS[compress]
        except KeyError:
            raise ConfigurationError(f"The {compress} compression format does not exist.")

    def write(self, artifact: T, dir: Union[str, PathLike]):
        filename = pathlib.Path(dir) / ("data.dill" + _SUFFIXES[self.open])
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
        filename = pathlib.Path(dir) / ("data.dill" + _SUFFIXES[self.open])
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
        self.f = _open_compressed(filename, "rb")
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


@Format.register("json")
class JsonFormat(Format[T], Generic[T]):
    """This format writes the artifact as a single file in json format.
    Optionally, it can compress the data. This is very flexible, but not always the fastest.

    This format has special support for iterables. If you write an iterator, it will consume the
    iterator. If you read an iterator, it will read the iterator lazily.
    """

    VERSION = 2

    def __init__(self, compress: Optional[str] = None):
        self.logger = cast(AllenNlpLogger, logging.getLogger(self.__class__.__name__))
        try:
            self.open = _OPEN_FUNCTIONS[compress]
        except KeyError:
            raise ConfigurationError(f"The {compress} compression format does not exist.")

    @staticmethod
    def _encoding_fallback(unencodable: Any):
        if isinstance(unencodable, torch.Tensor):
            if len(unencodable.shape) == 0:
                return unencodable.item()
            else:
                raise TypeError(
                    "Tensors must have 1 element and no dimensions to be JSON serializable."
                )
        elif dataclasses.is_dataclass(unencodable):
            result = dataclasses.asdict(unencodable)
            module = type(unencodable).__module__
            qualname = type(unencodable).__qualname__
            if module == "builtins":
                result["_dataclass"] = qualname
            else:
                result["_dataclass"] = [module, qualname]
            return result
        raise TypeError(f"Object of type {type(unencodable)} is not JSON serializable")

    @staticmethod
    def _decoding_fallback(o: Dict) -> Any:
        if "_dataclass" in o:
            classname: Union[str, List[str]] = o.pop("_dataclass")
            if isinstance(classname, list) and len(classname) == 2:
                module, classname = classname
                constructor: Callable = importlib.import_module(module)  # type: ignore
                for item in classname.split("."):
                    constructor = getattr(constructor, item)
            elif isinstance(classname, str):
                constructor = globals()[classname]
            else:
                raise RuntimeError(f"Could not parse {classname} as the name of a dataclass.")
            return constructor(**o)
        return o

    def write(self, artifact: T, dir: Union[str, PathLike]):
        if hasattr(artifact, "__next__"):
            filename = pathlib.Path(dir) / ("data.jsonl" + _SUFFIXES[self.open])
            with self.open(filename, "wt") as f:
                for item in cast(Iterable, artifact):
                    json.dump(item, f, default=self._encoding_fallback)
                    f.write("\n")
        else:
            filename = pathlib.Path(dir) / ("data.json" + _SUFFIXES[self.open])
            with self.open(filename, "wt") as f:
                json.dump(artifact, f, default=self._encoding_fallback)

    def read(self, dir: Union[str, PathLike]) -> T:
        iterator_filename = pathlib.Path(dir) / ("data.jsonl" + _SUFFIXES[self.open])
        iterator_exists = iterator_filename.exists()
        non_iterator_filename = pathlib.Path(dir) / ("data.json" + _SUFFIXES[self.open])
        non_iterator_exists = non_iterator_filename.exists()

        if iterator_exists and non_iterator_exists:
            self.logger.warning(
                "Both %s and %s exist. Ignoring %s.",
                iterator_filename,
                non_iterator_filename,
                iterator_filename,
            )
            iterator_exists = False

        if not iterator_exists and not non_iterator_exists:
            raise IOError("Attempting to read non-existing data from %s", dir)
        if iterator_exists and not non_iterator_exists:
            return JsonFormatIterator(iterator_filename)  # type: ignore
        elif not iterator_exists and non_iterator_exists:
            with self.open(non_iterator_filename, "rt") as f:
                return json.load(f, object_hook=self._decoding_fallback)
        else:
            raise RuntimeError("This should be impossible.")


class JsonFormatIterator(Iterator[T], Generic[T]):
    """This class is used so we can return an iterator from `JsonFormat.read()`."""

    def __init__(self, filename: Union[str, PathLike]):
        self.f = _open_compressed(filename, "rt")

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self.f is None:
            raise StopIteration()
        try:
            line = self.f.readline()
            if len(line) <= 0:
                raise EOFError()
            return json.loads(line, object_hook=JsonFormat._decoding_fallback)
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
