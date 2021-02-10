from os import PathLike
from typing import Dict, Iterator, Union

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader


@DatasetReader.register("multitask")
class MultiTaskDatasetReader(DatasetReader):
    """
    This `DatasetReader` simply collects a dictionary of other `DatasetReaders`.  It is designed for
    a different class (the `MultiTaskDataLoader`) to actually read from each of the underlying
    dataset readers, and so this really is just a glorified dictionary that we can construct as a
    `DatasetReader`.  We throw an error if you try to actually call `read()`, because you should be
    doing that differently.

    Registered as a `DatasetReader` with name "multitask".

    # Parameters

    readers : `Dict[str, DatasetReader]`
        A mapping from dataset name to `DatasetReader` objects for reading that dataset.  You can
        use whatever names you want for the datasets, but they have to match the keys you use for
        data files and in other places in the `MultiTaskDataLoader` and `MultiTaskScheduler`.
    """

    def __init__(self, readers: Dict[str, DatasetReader]) -> None:
        self.readers = readers

    def read(self, file_paths: Dict[str, Union[PathLike, str]]) -> Dict[str, Iterator[Instance]]:  # type: ignore
        raise RuntimeError("This class is not designed to be called like this")
