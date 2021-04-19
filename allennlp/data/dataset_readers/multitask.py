from os import PathLike
from typing import Dict, Iterator, Union, Optional

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import (
    DatasetReader,
    WorkerInfo,
    DatasetReaderInput,
)


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
        self.readers = {
            task: _MultitaskDatasetReaderShim(reader, task) for task, reader in readers.items()
        }

    def read(  # type: ignore
        self,
        file_paths: Union[PathLike, str, Dict[str, Union[PathLike, str]]],
        *,
        force_task: Optional[str] = None
    ) -> Union[Iterator[Instance], Dict[str, Iterator[Instance]]]:
        if force_task is None:
            raise RuntimeError("This class is not designed to be called like this.")
        return self.readers[force_task].read(file_paths)


@DatasetReader.register("multitask_shim")
class _MultitaskDatasetReaderShim(DatasetReader):
    """This dataset reader wraps another dataset reader and adds the name of the "task" into
    each instance as a metadata field. You should not have to use this yourself."""

    def __init__(self, inner: DatasetReader, head: str, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.head = head

    def _set_worker_info(self, info: Optional[WorkerInfo]) -> None:
        """
        Should only be used internally.
        """
        super()._set_worker_info(info)
        self.inner._set_worker_info(info)

    def read(self, file_path: DatasetReaderInput) -> Iterator[Instance]:
        from allennlp.data.fields import MetadataField

        for instance in self.inner.read(file_path):
            instance.add_field("task", MetadataField(self.head))
            yield instance

    def text_to_instance(self, *inputs) -> Instance:
        from allennlp.data.fields import MetadataField

        instance = self.inner.text_to_instance(*inputs)
        instance.add_field("task", MetadataField(self.head))
        return instance

    def apply_token_indexers(self, instance: Instance) -> None:
        self.inner.apply_token_indexers(instance)
