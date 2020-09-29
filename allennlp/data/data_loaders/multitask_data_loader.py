from collections import defaultdict
from typing import Any, Dict, Iterator

from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict, allennlp_collate
from allennlp.data.data_loaders.multi_process_data_loader import MultiProcessDataLoader
from allennlp.data.data_loaders.multitask_scheduler import MultiTaskScheduler
from allennlp.data.dataset_readers.multitask import MultiTaskDatasetReader
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary


@DataLoader.register("multitask")
class MultiTaskDataLoader(DataLoader):
    """
    Registered as a `DataLoader` with name "multitask".
    """

    def __init__(
        self,
        reader: MultiTaskDatasetReader,
        data_path: Dict[str, str],
        scheduler: MultiTaskScheduler,
        batch_size: int,
        *,
        batch_size_multiplier: Dict[str, int] = None,
        batches_per_epoch: int = None,
        drop_last: bool = False,
        num_workers: Dict[str, int] = None,
        max_instances_in_memory: Dict[str, int] = None,
        start_method: Dict[str, str] = None,
        instance_queue_size: Dict[str, int] = None,
        instance_chunk_size: Dict[str, int] = None,
    ) -> None:
        self.readers = reader.readers
        self.data_paths = data_path
        self.scheduler = scheduler

        self._batch_size = batch_size
        self._batches_per_epoch = batches_per_epoch
        self._batch_size_multiplier = batch_size_multiplier or defaultdict(lambda: 1)
        self._drop_last = drop_last

        self._num_workers = num_workers or {}
        self._max_instances_in_memory = max_instances_in_memory or {}
        self._start_method = start_method or {}
        self._instance_queue_size = instance_queue_size or {}
        self._instance_chunk_size = instance_chunk_size or {}

        if self.readers.keys() != self.data_paths.keys():
            raise ValueError(
                f"Mismatch between readers ({self.readers.keys()}) and data paths"
                " ({self.data_paths.keys()})"
            )
        self._loaders = {key: self._make_data_loader(key) for key in self.readers}

    def __len__(self) -> int:
        if self._batches_per_epoch is not None:
            return self._batches_per_epoch

        # Here we try to estimate the actual length.  If you are using varying batch size
        # multipliers per task, we may get batch packing orders that make this an underestimate, as
        # this assumes that all batches are full, which may not be true.
        total_instances = 0
        for key, loader in self._loaders.items():
            # This will raise a TypeError if any of the underlying loaders doesn't have a length,
            # which is actually what we want.  If the loader has a length, we set batch_size = 1, so
            # this will give us the right number of instances.
            total_instances += self._batch_size_multiplier[key] * len(loader)
        if self._drop_last or total_instances % self._batch_size == 0:
            return total_instances // self._batch_size
        else:
            return 1 + total_instances // self._batch_size

    def __iter__(self) -> Iterator[TensorDict]:
        # Basic outline: first we _sample_ the instances that we're going to be using for this
        # epoch, which relies on the scheduler if `self._batches_per_epoch` is not None.  This is
        # basically just saying how many instances we should use this epoch for each dataset, and we
        # grab bounded-length iterators over that many instances for each dataset.  Second, we
        # _schedule_ the epoch's instances into batches, again relying on the scheduler.
        epoch_generators = self._get_instances_for_epoch()

    def iter_instances(self) -> Iterator[Instance]:
        # The only external contract for this method is that it iterates over instances
        # individually; it doesn't actually specify anything about batching or anything else.  The
        # implication is that you iterate over all instances in the dataset, in an arbitrary order.
        # The only external uses of this method are in vocabulary construction (the
        # MultiProcessDataLoader uses this function internally when constructing batches, but that's
        # an implementation detail).
        #
        # So, the only thing we need to do here is iterate over all instances from all datasets, and
        # that's sufficient.  We won't be using this for batching, because that requires some
        # complex, configurable scheduling.
        #
        # The underlying data loaders here could be using multiprocessing; we don't need to worry
        # about that in this class.  Caching is also handled by the underlying data loaders.
        for loader in self._loaders.values():
            yield from loader.iter_instances()

    def index_with(self, vocab: Vocabulary) -> None:
        for loader in self._loaders.values():
            loader.index_with(vocab)

    def _make_data_loader(self, key: str) -> MultiProcessDataLoader:
        kwargs: Dict[str, Any] = {}
        kwargs["reader"] = self.readers[key]
        kwargs["data_path"] = self.data_paths[key]
        kwargs["batch_size"] = 1
        if key in self._num_workers:
            kwargs["num_workers"] = self._num_workers[key]
        if key in self._max_instances_in_memory:
            kwargs["max_instances_in_memory"] = self._max_instances_in_memory[key]
        if key in self._start_method:
            kwargs["start_method"] = self._start_method[key]
        if key in self._instance_queue_size:
            kwargs["instance_queue_size"] = self._instance_queue_size[key]
        if key in self._instance_chunk_size:
            kwargs["instance_chunk_size"] = self._instance_chunk_size[key]
        return MultiProcessDataLoader(**kwargs)
