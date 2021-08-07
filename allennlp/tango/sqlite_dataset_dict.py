import gzip
import string
from os import PathLike
from pathlib import Path
from typing import (
    Union,
    Any,
    Mapping,
    Optional,
    Sequence,
    MutableSequence,
    MutableMapping,
    Iterator,
    Set,
    Dict,
)

import dill
from sqlitedict import SqliteDict

from allennlp.data import Vocabulary
from allennlp.tango.dataset import DatasetDictBase


class SqliteSparseSequence(MutableSequence[Any]):
    def __init__(
        self,
        filename: Union[str, PathLike],
        read_only: bool = False
    ):
        self.table = SqliteDict(filename, flag="r" if read_only else "c")

    def __del__(self):
        self.close()

    def __getitem__(self, i: Union[int, slice]) -> Any:
        if isinstance(i, int):
            try:
                return self.table[str(i)]
            except KeyError:
                current_length = len(self)
                if i >= current_length or current_length <= 0:
                    raise IndexError("list index out of range")
                elif i < 0 < current_length:
                    return self.__getitem__(i % current_length)
                else:
                    return None
        elif isinstance(i, slice):
            from allennlp.tango.dataloader import ShuffledSequence

            return ShuffledSequence(self, i.indices(len(self)))
        else:
            raise TypeError(f"list indices must be integers or slices, not {i.__class__.__name__}")

    def __setitem__(self, i: Union[int, slice], value: Any):
        if isinstance(i, int):
            current_length = len(self)
            if i < 0:
                i %= current_length
            self.table[str(i)] = value
            self.table["_len"] = max(i, current_length)
            self.table.commit()
        else:
            raise TypeError(f"list indices must be integers, not {i.__class__.__name__}")

    def __delitem__(self, i: Union[int, slice]):
        current_length = len(self)
        if isinstance(i, int):
            if i < 0:
                i %= current_length
            if i >= current_length:
                raise IndexError("list assignment index out of range")
            for index in range(i + 1, current_length):
                self.table[str(index - 1)] = self.table.get(str(index))
            del self.table[str(current_length - 1)]
            self.table["_len"] = current_length - 1
            self.table.commit()
        elif isinstance(i, slice):
            for index in i.indices(current_length):
                del self[index]
        else:
            raise TypeError(f"list indices must be integers or slices, not {i.__class__.__name__}")

    def insert(self, i: int, value: Any) -> None:
        current_length = len(self)
        for index in reversed(range(i, current_length)):
            self.table[str(index + 1)] = self.table.get(str(index))
        self.table[str(i)] = value
        self.table["_len"] = current_length + 1
        self.table.commit()

    def __len__(self) -> int:
        try:
            return self.table["_len"]
        except KeyError:
            return 0

    def clear(self) -> None:
        self.table.clear()
        self.table.commit()

    def close(self) -> None:
        if self.table is not None:
            self.table.close()
            self.table = None


class SqliteDatasetDict(DatasetDictBase, MutableMapping[str, MutableSequence[Any]]):
    SAFE_FILENAME_CHARS = validFilenameChars = frozenset("-_.%s%s" % (string.ascii_letters, string.digits))

    @classmethod
    def filename_is_safe(cls, filename: str) -> bool:
        return all(c in cls.SAFE_FILENAME_CHARS for c in filename)

    def __init__(self, directory: Union[str, PathLike], read_only: bool = False):
        self.directory = Path(directory)
        self.read_only = read_only
        self.metadata_table = SqliteDict(self.directory / "_metadata.sqlite", flag="r" if read_only else "c", autocommit=True)
        self._split_sequence_cache: Dict[str, SqliteSparseSequence] = {}

    def __del__(self):
        self.close()

    def _filename_for_split(self, split_name: str) -> Path:
        if not self.filename_is_safe(split_name):
            raise ValueError(f"{split_name} contains invalid characters")
        return self.directory / f"{split_name}.sqlite"

    @property
    def metadata(self) -> MutableMapping[str, Any]:
        return self.metadata_table

    @metadata.setter
    def metadata(self, new_metadata: Mapping[str, Any]):
        self.metadata_table.clear()
        self.metadata_table.update(new_metadata)

    @property
    def vocab(self) -> Optional[Vocabulary]:
        try:
            with gzip.open(self.directory / "vocab.dill.gz", "rb") as f:
                return dill.load(f)
        except FileNotFoundError:
            return None

    @vocab.setter
    def vocab(self, new_vocab: Vocabulary):
        with gzip.open(self.directory / "vocab.dill.gz", "wb") as f:
            dill.dump(new_vocab, f)

    @property
    def splits(self) -> MutableMapping[str, MutableSequence[Any]]:
        return self

    @splits.setter
    def splits(self, new_splits: Mapping[str, Sequence[Any]]):
        for split_name, sequence in new_splits.items():
            self[split_name] = sequence
        for split_name in self.keys():
            if split_name not in new_splits:
                del self[split_name]

    def __setitem__(self, split_name: str, items: Sequence[Any]) -> None:
        if self.read_only:
            raise ValueError("Cannot set items on read-only collections")
        split_filename = self._filename_for_split(split_name)
        split_filename.unlink(missing_ok=True)

        sqlite_sequence = SqliteSparseSequence(split_filename, self.read_only)
        sqlite_sequence.extend(items)
        self._split_sequence_cache[split_name] = sqlite_sequence

    def __delitem__(self, split_name: str) -> None:
        if self.read_only:
            raise ValueError("Cannot delete items on read-only collections")

        try:
            del self._split_sequence_cache[split_name]
        except KeyError:
            pass

        split_filename = self._filename_for_split(split_name)
        try:
            split_filename.unlink(missing_ok=False)
        except FileNotFoundError:
            raise KeyError(split_name)

    def __getitem__(self, split_name: str) -> MutableSequence[Any]:
        try:
            return self._split_sequence_cache[split_name]
        except KeyError:
            split_filename = self._filename_for_split(split_name)
            result = SqliteSparseSequence(split_filename, self.read_only)
            self._split_sequence_cache[split_name] = result
            return result

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def __iter__(self) -> Iterator[str]:
        for path in self.directory.glob("*.sqlite"):
            split_name = str(path.stem)
            if not split_name.startswith("_"):
                yield split_name

    def close(self):
        for sequence in self._split_sequence_cache.values():
            sequence.close()
        self._split_sequence_cache.clear()

        if self.metadata_table is not None:
            self.metadata_table.close()
            self.metadata_table = None
