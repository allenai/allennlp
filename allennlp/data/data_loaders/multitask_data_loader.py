from typing import Any, Dict, Iterable, Iterator, List
import itertools
import math

from allennlp.common import util
from allennlp.data.batch import Batch
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.data.data_loaders.multi_process_data_loader import MultiProcessDataLoader
from allennlp.data.data_loaders.multitask_scheduler import (
    MultiTaskScheduler,
    HomogeneousRoundRobinScheduler,
)
from allennlp.data.data_loaders.multitask_epoch_sampler import MultiTaskEpochSampler
from allennlp.data.dataset_readers.multitask import MultiTaskDatasetReader
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary


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
    batch_size: `int`
        The number of instances (from any dataset) that should be combined together into a single
        batch.  See also the `batch_size_multiplier` argument for additional control over exactly
        how batch size is computed.
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
    batch_size_multiplier: `Dict[str, float]`, optional (default = `None`)
        If this is not `None`, it specifies how much of the batch an instance from each dataset
        takes up.  That is, if this is 1 for every dataset (which is the default), then batch size
        is computed as normal.  If dataset "A" has a value of 1.5 in this dictionary, than each
        instance from dataset "A" counts as 1.5 instances for the purposes of computing batch size.
        This option is available to you to account for the fact that some operations might be *much*
        less costly than others (e.g., if you are multitasking a coref model with a simple document
        classification model).  If you use this, you're on your own as far as figuring out how it
        interacts with optimization behavior.
    instances_per_epoch: `int`, optional (default = `None`)
        If not `None`, we will use this many instances per epoch of training, drawing from the
        underlying datasets with proportions given by the `scheduler`.  Note that this is
        _instances_, not _batches_, because if you're using batch size multipliers we don't know how
        many batches the instances specified by the `scheduler` will turn out to be.
    drop_last: `bool`, optional (default = `False`)
        If this is `True`, we will not return the last batch if it is smaller than `batch_size`.
        Note that this is kind of nonsensical to use if you're using `batch_size_multipliers`, as
        you are not guaranteed to get an optimal packing, so you will likely have batches that don't
        fill up the `batch_size` in that case, anyway.
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
    """

    def __init__(
        self,
        reader: MultiTaskDatasetReader,
        data_path: Dict[str, str],
        batch_size: int,
        *,
        scheduler: MultiTaskScheduler = None,
        sampler: MultiTaskEpochSampler = None,
        instances_per_epoch: int = None,
        batch_size_multiplier: Dict[str, float] = None,
        drop_last: bool = False,
        num_workers: Dict[str, int] = None,
        max_instances_in_memory: Dict[str, int] = None,
        start_method: Dict[str, str] = None,
        instance_queue_size: Dict[str, int] = None,
        instance_chunk_size: Dict[str, int] = None,
        shuffle: bool = True,
    ) -> None:
        self.readers = reader.readers
        self.data_paths = data_path
        self.scheduler = scheduler or HomogeneousRoundRobinScheduler(batch_size=batch_size)
        self.sampler = sampler

        self._batch_size = batch_size
        self._instances_per_epoch = instances_per_epoch
        self._batch_size_multiplier = batch_size_multiplier or {}
        for multiplier in self._batch_size_multiplier.values():
            if multiplier > batch_size:
                raise ValueError(
                    f"Multiplier value ({multiplier}) is larger than batch size ({batch_size})"
                )
        self._drop_last = drop_last
        self._shuffle = shuffle

        if instances_per_epoch is not None and sampler is None:
            raise ValueError(
                "You must provide an EpochSampler if you want to not use all instances every epoch"
            )

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
                # iteration...  mypy also doesn't know what to do with this, for some reason I can't
                # figure out.
                lambda l=loader: maybe_shuffle_instances(l, self._shuffle)  # type: ignore
            )
            for key, loader in self._loaders.items()
        }

    def __len__(self) -> int:
        if self._instances_per_epoch is not None:
            return self._instances_per_epoch

        # Here we try to estimate the actual length.  If you are using varying batch size
        # multipliers per task, we may get batch packing orders that make this an underestimate, as
        # this assumes that all batches are full, which may not be true.
        total_instances = 0.0
        for key, loader in self._loaders.items():
            # This will raise a TypeError if any of the underlying loaders doesn't have a length,
            # which is actually what we want.  If the loader has a length, we set batch_size = 1, so
            # this will give us the right number of instances.
            total_instances += self._batch_size_multiplier.get(key, 1.0) * len(loader)
        if self._drop_last or total_instances % self._batch_size == 0:
            return int(total_instances) // self._batch_size
        else:
            return int(1 + total_instances) // self._batch_size

    def __iter__(self) -> Iterator[TensorDict]:
        # Basic outline: first we _sample_ the instances that we're going to be using for this
        # epoch, which relies on the scheduler if `self._instances_per_epoch` is not None.  This is
        # basically just saying how many instances we should use this epoch for each dataset, and we
        # grab bounded-length iterators over that many instances for each dataset.  Second, we
        # _schedule_ the epoch's instances into a single list, again relying on the scheduler.
        # Finally, we take that combined list and yield `batch_size` batches from it.
        epoch_instances = self._get_instances_for_epoch()
        scheduled_instances = self.scheduler.order_epoch_instances(epoch_instances)
        batch_instances: List[Instance] = []
        current_batch_size = 0.0
        for dataset, instance in scheduled_instances:
            current_batch_size += self._batch_size_multiplier.get(dataset, 1.0)
            if current_batch_size > self._batch_size:
                batch = Batch(batch_instances)
                yield batch.as_tensor_dict()
                batch_instances = [instance]
                current_batch_size = self._batch_size_multiplier.get(dataset, 1.0)
            else:
                batch_instances.append(instance)

        # Based on how we yield batches above, we are guaranteed to always have leftover instances,
        # so we don't need a check for that here.
        if not self._drop_last or current_batch_size == self._batch_size:
            batch = Batch(batch_instances)
            yield batch.as_tensor_dict()

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

    def _get_instances_for_epoch(self) -> Dict[str, Iterable[Instance]]:
        if self._instances_per_epoch is None:
            return {
                key: maybe_shuffle_instances(loader, self._shuffle)
                for key, loader in self._loaders.items()
            }
        if self.sampler is None:
            # We already checked for this in the constructor, so this should never happen unless you
            # modified the object after creation.  But mypy is complaining, so here's another check.
            raise ValueError(
                "You must specify an EpochSampler if self._instances_per_epoch is not None"
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
