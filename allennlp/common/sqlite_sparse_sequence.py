import os
import shutil
from os import PathLike
from typing import MutableSequence, Any, Union, Iterable

from sqlitedict import SqliteDict

from allennlp.tango.dataloader import ShuffledSequence


class SqliteSparseSequence(MutableSequence[Any]):
    def __init__(self, filename: Union[str, PathLike], read_only: bool = False):
        self.table = SqliteDict(filename, "sparse_sequence", flag="r" if read_only else "c")

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
            return ShuffledSequence(self, range(*i.indices(len(self))))
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
            # This isn't very efficient for continuous slices.
            for index in reversed(range(*i.indices(current_length))):
                del self[index]
        else:
            raise TypeError(f"list indices must be integers or slices, not {i.__class__.__name__}")

    def extend(self, values: Iterable[Any]) -> None:
        current_length = len(self)
        for index, value in enumerate(values):
            self.table[str(index + current_length)] = value
        self.table["_len"] = current_length + index + 1
        self.table.commit()

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

    def copy_to(self, target: Union[str, PathLike]):
        try:
            os.link(self.table.filename, target)
        except OSError as e:
            if e.errno == 18:  # Cross-device link
                shutil.copy(self.table.filename, target)
            else:
                raise
