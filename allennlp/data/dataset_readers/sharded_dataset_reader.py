import glob
import logging
import torch
from typing import Iterable

from allennlp.common import util
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance


logger = logging.getLogger(__name__)


@DatasetReader.register("sharded")
class ShardedDatasetReader(DatasetReader):
    """
    Wraps another dataset reader and uses it to read from multiple input files.
    Note that in this case the `file_path` passed to `read()` should be a glob,
    and that the dataset reader will return instances from all files matching
    the glob.

    The order the files are processed in is deterministic to enable the
    instances to be filtered according to worker rank in the distributed case.

    Registered as a `DatasetReader` with name "sharded".

    # Parameters

    base_reader : `DatasetReader`
        Reader with a read method that accepts a single file.
    """

    def __init__(self, base_reader: DatasetReader, **kwargs,) -> None:
        super().__init__(**kwargs)

        if util.is_distributed():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        self.reader = base_reader
        self.reader.manual_distributed_sharding = True

    def text_to_instance(self, *args, **kwargs) -> Instance:
        """
        Just delegate to the base reader text_to_instance.
        """
        return self.reader.text_to_instance(*args, **kwargs)  # type: ignore

    def _read(self, file_path: str) -> Iterable[Instance]:
        shards = glob.glob(file_path)
        # Ensure a consistent order.
        shards.sort()

        for i, shard in enumerate(shards):
            if i % self._world_size == self._rank:
                logger.info(f"reading instances from {shard}")
                for instance in self.reader.read(shard):
                    yield instance
