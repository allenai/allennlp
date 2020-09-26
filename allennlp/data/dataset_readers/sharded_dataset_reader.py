import glob
import logging
import os
import torch
from typing import Iterable

from allennlp.common import util
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance


logger = logging.getLogger(__name__)


@DatasetReader.register("sharded")
class ShardedDatasetReader(DatasetReader):
    """
    Wraps another dataset reader and uses it to read from multiple input files.

    Note that in this case the `file_path` passed to `read()` should either be a glob path
    or a path or URL to an archive file ('.zip' or '.tar.gz').

    The dataset reader will return instances from all files matching the glob, or all
    files within the archive.

    The order the files are processed in is deterministic to enable the
    instances to be filtered according to worker rank in the distributed case.

    Registered as a `DatasetReader` with name "sharded".

    # Parameters

    base_reader : `DatasetReader`
        Reader with a read method that accepts a single file.
    """

    def __init__(self, base_reader: DatasetReader, **kwargs) -> None:
        super().__init__(manual_distributed_sharding=True, **kwargs)

        if util.is_distributed():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        self.reader = base_reader
        # We have to check that the base reader doesn't implement manual distributed
        # sharding itself, because if it does, then only a fraction of the instances
        # will be read.
        if getattr(self.reader, "manual_distributed_sharding", False):
            raise ValueError(
                "The base reader of a sharded dataset reader should not implement "
                "manual distributed sharding itself."
            )
        # However we still need to set this flag to `True` after the fact so that
        # all of the instances within each shard are used.
        self.reader.manual_distributed_sharding = True

    def text_to_instance(self, *args, **kwargs) -> Instance:
        """
        Just delegate to the base reader text_to_instance.
        """
        return self.reader.text_to_instance(*args, **kwargs)  # type: ignore

    def _read(self, file_path: str) -> Iterable[Instance]:
        try:
            maybe_extracted_archive = cached_path(file_path, extract_archive=True)
            if not os.path.isdir(maybe_extracted_archive):
                # This isn't a directory, so `file_path` is just a file.
                raise ConfigurationError(f"{file_path} should be an archive or directory")
            shards = [
                os.path.join(maybe_extracted_archive, p)
                for p in os.listdir(maybe_extracted_archive)
                if not p.startswith(".")
            ]
            if not shards:
                raise ConfigurationError(f"No files found in {file_path}")
        except FileNotFoundError:
            # Not a local or remote archive, so treat as a glob.
            shards = glob.glob(file_path)
            if not shards:
                raise ConfigurationError(f"No files found matching {file_path}")

        # Ensure a consistent order.
        shards.sort()

        for i, shard in enumerate(shards):
            if i % self._world_size == self._rank:
                logger.info(f"reading instances from {shard}")
                for instance in self.reader.read(shard):
                    yield instance
