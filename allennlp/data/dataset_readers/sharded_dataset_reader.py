import glob
import logging
import os
from typing import Iterable

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, PathOrStr
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
    instances to be filtered according to worker rank in the distributed training or multi-process
    data loading scenarios. In either case, the number of file shards should ideally be a multiple
    of the number of workers, and each file should produce roughly the same number of instances.

    Registered as a `DatasetReader` with name "sharded".

    # Parameters

    base_reader : `DatasetReader`
        Reader with a read method that accepts a single file.
    """

    def __init__(self, base_reader: DatasetReader, **kwargs) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self.reader = base_reader
        # We have to make the base reader think that it's the only worker so that it doesn't
        # do any of its own filtering.
        self.reader._set_worker_info(None)
        self.reader._set_distributed_info(None)

    def text_to_instance(self, *args, **kwargs) -> Instance:
        """
        Just delegate to the base reader text_to_instance.
        """
        return self.reader.text_to_instance(*args, **kwargs)  # type: ignore

    def _read(self, file_path: PathOrStr) -> Iterable[Instance]:
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
            shards = glob.glob(str(file_path))
            if not shards:
                raise ConfigurationError(f"No files found matching {file_path}")

        # Ensure a consistent order.
        shards.sort()

        for shard in self.shard_iterable(shards):
            logger.info(f"reading instances from {shard}")
            for instance in self.reader.read(shard):
                yield instance
