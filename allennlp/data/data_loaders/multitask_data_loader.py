from typing import Any, Dict, Iterable, Iterator, Union, Optional
import itertools
import math

import torch
from overrides import overrides

from allennlp.common import util
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, DatasetReaderInput
from allennlp.data.batch import Batch
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.data_loaders.multitask_scheduler import MultiTaskScheduler
from allennlp.data.data_loaders.multitask_epoch_sampler import MultiTaskEpochSampler
from allennlp.data.dataset_readers.multitask import MultiTaskDatasetReader
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as nn_util


def maybe_shuffle_instances(loader: DataLoader, shuffle: bool) -> Iterable[Instance]:
    if shuffle:
        return util.shuffle_iterable(loader.iter_instances())
    else:
        return loader.iter_instances()


@DataLoader.register("multitask")
class MultiTaskDataLoader(DataLoader):
    """
    A `DataLoader` intended for multi-task learning.  The basic idea is that you use a
    `MultiTaskDatasetReader`, which takes a dictionary of `DatasetReaders`, keyed by some name.  You
    use those same names for various parameters here, including the data paths that get passed to
    each reader.  We will load each dataset and iterate over instances in them using a
    `MultiTaskEpochSampler` and a `MultiTaskScheduler`.  The `EpochSampler` says how much to use
    from each dataset at each epoch, and the `Scheduler` orders the instances in the epoch however
    you want.  Both of these are designed to be used in conjunction with trainer `Callbacks`, if
    desired, to have the sampling and/or scheduling behavior be dependent on the current state of
    training.

    While it is not necessarily required, this `DatasetReader` was designed to be used alongside a
    `MultiTaskModel`, which can handle instances coming from different datasets.  If your datasets
    are similar enough (say, they are all reading comprehension datasets with the same format), or
    your model is flexible enough, then you could feasibly use this `DataLoader` with a normal,
    non-multitask `Model`.

    Registered as a `DataLoader` with name "multitask".

    # Parameters

    reader: `MultiTaskDatasetReader`
    data_path: `Dict[str, str]`
        One file per underlying dataset reader in the `MultiTaskDatasetReader`, which will be passed
        to those readers to construct one `DataLoader` per dataset.
    scheduler: `MultiTaskScheduler`, optional (default = `HomogeneousRoundRobinScheduler`)
        The `scheduler` determines how instances are ordered within an epoch.  By default, we'll
        select one batch of instances from each dataset in turn, trying to ensure as uniform a mix
        of datasets as possible.  Note that if your model can handle it, using a
        `RoundRobinScheduler` is likely better than a `HomogeneousRoundRobinScheduler` (because it
        does a better job mixing gradient signals from various datasets), so you may want to
        consider switching.  We use the homogeneous version as default because it should work for
        any allennlp model, while the non-homogeneous one might not.
    sampler: `MultiTaskEpochSampler`, optional (default = `None`)
        Only used if `instances_per_epoch` is not `None`. If we need to select a subset of the data
        for an epoch, this `sampler` will tell us with what proportion we should sample from each
        dataset.  For instance, we might want to focus more on datasets that are underperforming in
        some way, by having those datasets contribute more instances this epoch than other datasets.
    instances_per_epoch: `int`, optional (default = `None`)
        If not `None`, we will use this many instances per epoch of training, drawing from the
        underlying datasets according to the `sampler`.
    num_workers: `Dict[str, int]`, optional (default = `None`)
        Used when creating one `MultiProcessDataLoader` per dataset.  If you want non-default
        behavior for this parameter in the `DataLoader` for a particular dataset, pass the
        corresponding value here, keyed by the dataset name.
    max_instances_in_memory: `Dict[str, int]`, optional (default = `None`)
        Used when creating one `MultiProcessDataLoader` per dataset.  If you want non-default
        behavior for this parameter in the `DataLoader` for a particular dataset, pass the
        corresponding value here, keyed by the dataset name.
    start_method: `Dict[str, str]`, optional (default = `None`)
        Used when creating one `MultiProcessDataLoader` per dataset.  If you want non-default
        behavior for this parameter in the `DataLoader` for a particular dataset, pass the
        corresponding value here, keyed by the dataset name.
    instance_queue_size: `Dict[str, int]`, optional (default = `None`)
        Used when creating one `MultiProcessDataLoader` per dataset.  If you want non-default
        behavior for this parameter in the `DataLoader` for a particular dataset, pass the
        corresponding value here, keyed by the dataset name.
    instance_chunk_size: `Dict[str, int]`, optional (default = `None`)
        Used when creating one `MultiProcessDataLoader` per dataset.  If you want non-default
        behavior for this parameter in the `DataLoader` for a particular dataset, pass the
        corresponding value here, keyed by the dataset name.
    shuffle: `bool`, optional (default = `True`)
        If `False`, we will not shuffle the instances that come from each underlying data loader.
        You almost certainly never want to use this except when debugging.
    cuda_device: `Optional[Union[int, str, torch.device]]`, optional (default = `None`)
        If given, batches will automatically be put on this device.

        !!! Note
            This should typically not be set in an AllenNLP configuration file. The `Trainer`
            will automatically call [`set_target_device()`](#set_target_device) before iterating
            over batches.
    """

    def __init__(
        self,
        reader: MultiTaskDatasetReader,
        data_path: Dict[str, str],
        scheduler: MultiTaskScheduler,
        *,
        sampler: MultiTaskEpochSampler = None,
        instances_per_epoch: int = None,
        num_workers: Dict[str, int] = None,
        max_instances_in_memory: Dict[str, int] = None,
        start_method: Dict[str, str] = None,
        instance_queue_size: Dict[str, int] = None,
        instance_chunk_size: Dict[str, int] = None,
        shuffle: bool = True,
        cuda_device: Optional[Union[int, str, torch.device]] = None,
    ) -> None:
        self.readers = reader.readers
        self.data_paths = data_path
        self.scheduler = scheduler
        self.sampler = sampler
        self.cuda_device: Optional[torch.device] = None
        if cuda_device is not None:
            if not isinstance(cuda_device, torch.device):
                self.cuda_device = torch.device(cuda_device)
            else:
                self.cuda_device = cuda_device

        self._instances_per_epoch = instances_per_epoch
        self._shuffle = shuffle

        if instances_per_epoch is not None and sampler is None:
            raise ValueError(
                "You must provide an EpochSampler if you want to not use all instances every epoch."
            )

        self._num_workers = num_workers or {}
        self._max_instances_in_memory = max_instances_in_memory or {}
        self._start_method = start_method or {}
        self._instance_queue_size = instance_queue_size or {}
        self._instance_chunk_size = instance_chunk_size or {}

        if self.readers.keys() != self.data_paths.keys():
            raise ValueError(
                f"Mismatch between readers ({self.readers.keys()}) and data paths "
                f"({self.data_paths.keys()})"
            )
        self._loaders = {key: self._make_data_loader(key) for key in self.readers}

        # This stores our current iterator with each dataset, so we don't just iterate over the
        # first k instances every epoch if we're using instances_per_epoch.  We'll grab instances
        # from here each epoch, and refresh it when it runs out.  We only use this in the case that
        # instances_per_epoch is not None, but these iterators are lazy, so always creating them
        # doesn't hurt anything.
        self._iterators: Dict[str, Iterator[Instance]] = {
            # NOTE: The order in which we're calling these iterator functions is important.  We want
            # an infinite iterator over the data, but we want the order in which we iterate over the
            # data to be different at every epoch.  The cycle function will give us an infinite
            # iterator, and it will call the lambda function each time it runs out of instances,
            # which will produce a new shuffling of the dataset.
            key: util.cycle_iterator_function(
                # This default argument to the lambda function is necessary to create a new scope
                # for the loader variable, so a _different_ loader gets saved for every iterator.
                # Dictionary comprehensions don't create new scopes in python.  If you don't have
                # this loader, you end up with `loader` always referring to the last loader in the
                # iteration... mypy also doesn't know what to do with this, for some reason I can't
                # figure out.
                lambda l=loader: maybe_shuffle_instances(l, self._shuffle)  # type: ignore
            )
            for key, loader in self._loaders.items()
        }

    @overrides
    def __len__(self) -> int:
        if self._instances_per_epoch is None:
            # This will raise a TypeError if any of the underlying loaders doesn't have a length,
            # which is actually what we want.
            return self.scheduler.count_batches(
                {dataset: len(loader) for dataset, loader in self._loaders.items()}
            )
        else:
            return self.scheduler.count_batches(
                {dataset: self._instances_per_epoch for dataset in self._loaders.keys()}
            )

    @overrides
    def __iter__(self) -> Iterator[TensorDict]:
        epoch_instances = self._get_instances_for_epoch()
        return (
            nn_util.move_to_device(
                Batch(instances).as_tensor_dict(),
                -1 if self.cuda_device is None else self.cuda_device,
            )
            for instances in self.scheduler.batch_instances(epoch_instances)
        )

    @overrides
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
        # about that in this class. Caching is also handled by the underlying data loaders.
        for loader in self._loaders.values():
            yield from loader.iter_instances()

    @overrides
    def index_with(self, vocab: Vocabulary) -> None:
        for loader in self._loaders.values():
            loader.index_with(vocab)

    @overrides
    def set_target_device(self, device: torch.device) -> None:
        self.cuda_device = device

    def _get_instances_for_epoch(self) -> Dict[str, Iterable[Instance]]:
        if self._instances_per_epoch is None:
            return {
                key: maybe_shuffle_instances(loader, self._shuffle)
                for key, loader in self._loaders.items()
            }
        if self.sampler is None:
            # We already checked for this in the constructor, so this should never happen unless you
            # modified the object after creation. But mypy is complaining, so here's another check.
            raise ValueError(
                "You must specify an EpochSampler if self._instances_per_epoch is not None."
            )
        dataset_proportions = self.sampler.get_task_proportions(self._loaders)
        proportion_sum = sum(dataset_proportions.values())
        num_instances_per_dataset = {
            key: math.floor(proportion * self._instances_per_epoch / proportion_sum)
            for key, proportion in dataset_proportions.items()
        }
        return {
            key: itertools.islice(self._iterators[key], num_instances)
            for key, num_instances in num_instances_per_dataset.items()
        }

    def _make_data_loader(self, key: str) -> MultiProcessDataLoader:
        kwargs: Dict[str, Any] = {
            "reader": _MultitaskDatasetReaderShim(self.readers[key], key),
            "data_path": self.data_paths[key],
            # We don't load batches from this data loader, only instances, but we have to set
            # something for the batch size, so we set 1.
            "batch_size": 1,
        }
        if key in self._num_workers:
            kwargs["num_workers"] = self._num_workers[key]
        if key in self._max_instances_in_memory:
            kwargs["max_instances_in_memory"] = self._max_instances_in_memory[key]
        if key in self._start_method:
            kwargs["start_method"] = self._start_method[key]
        return MultiProcessDataLoader(**kwargs)


@DatasetReader.register("multitask_shim")
class _MultitaskDatasetReaderShim(DatasetReader):
    """This dataset reader wraps another dataset reader and adds the name of the "task" into
    each instance as a metadata field. This exists only to support `MultitaskDataLoader`. You
    should not have to use this yourself."""

    def __init__(self, inner: DatasetReader, head: str, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.head = head

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
