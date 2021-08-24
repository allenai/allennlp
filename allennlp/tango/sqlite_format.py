import gzip
import pathlib
from os import PathLike
from typing import Union

import dill

from allennlp.common.file_utils import filename_is_safe
from allennlp.common.sqlite_sparse_sequence import SqliteSparseSequence
from allennlp.tango.dataset import DatasetDict
from allennlp.tango.format import Format


@Format.register("sqlite")
class SqliteDictFormat(Format[DatasetDict]):
    VERSION = 2

    def write(self, artifact: DatasetDict, dir: Union[str, PathLike]):
        dir = pathlib.Path(dir)
        with gzip.open(dir / "vocab.dill.gz", "wb") as f:
            dill.dump(artifact.vocab, f)
        with gzip.open(dir / "metadata.dill.gz", "wb") as f:
            dill.dump(artifact.metadata, f)
        for split_name, split in artifact.splits.items():
            filename = f"{split_name}.sqlite"
            if not filename_is_safe(filename):
                raise ValueError(f"{split_name} is not a valid name for a split.")
            if isinstance(split, SqliteSparseSequence):
                split.copy_to(filename)
            else:
                (dir / filename).unlink(missing_ok=True)
                sqlite = SqliteSparseSequence(dir / filename)
                sqlite.extend(split)

    def read(self, dir: Union[str, PathLike]) -> DatasetDict:
        dir = pathlib.Path(dir)
        with gzip.open(dir / "vocab.dill.gz", "rb") as f:
            vocab = dill.load(f)
        with gzip.open(dir / "metadata.dill.gz", "rb") as f:
            metadata = dill.load(f)
        splits = {
            filename.stem: SqliteSparseSequence(filename, read_only=True)
            for filename in dir.glob("*.sqlite")
        }
        return DatasetDict(vocab=vocab, metadata=metadata, splits=splits)
