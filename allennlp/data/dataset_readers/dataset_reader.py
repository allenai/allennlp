from dataclasses import dataclass
import itertools
from os import PathLike
from typing import Iterable, Iterator, Optional, Union, TypeVar
import logging
import warnings

import torch.distributed as dist

from allennlp.data.instance import Instance
from allennlp.common import util
from allennlp.common.registrable import Registrable


logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """
    Contains information about the worker context when a `DatasetReader`
    is being used within a multi-process `DataLoader`.

    From a `DatasetReader` this can accessed with the [`get_worker_info()`](#get_worker_info) method.
    """

    num_workers: int
    """
    The total number of workers.
    """

    id: int
    """
    The 0-indexed ID of the current worker.
    """


_T = TypeVar("_T")


class DatasetReader(Registrable):
    """
    A `DatasetReader` knows how to turn a file containing a dataset into a collection
    of `Instance`s.  To implement your own, just override the [`_read(file_path)`](#_read) method
    to return an `Iterable` of the instances. Ideally this should be a lazy generator
    that yields them one at a time.

    All parameters necessary to `_read` the data apart from the filepath should be passed
    to the constructor of the `DatasetReader`.

    You should also implement [`text_to_instance(*inputs)`](#text_to_instance),
    which should be used to turn raw data into `Instance`s. This method is required
    in order to use a `Predictor` with your reader.

    Usually the `_read()` method is implemented to call `text_to_instance()`.

    # Parameters

    max_instances : `int`, optional (default=`None`)
        If given, will stop reading after this many instances. This is a useful setting for debugging.
        Setting this disables caching.

    manual_distributed_sharding: `bool`, optional (default=`False`)
        By default, when used in a distributed setting, `DatasetReader` makes sure that each
        worker process only receives a subset of the data. It does this by reading the whole
        dataset in each worker, but filtering out the instances that are not needed.

        While this ensures that each worker will recieve unique instances, it's not a very efficient
        way to do so since each worker still needs to process every single instance.

        A better way to handle this is to manually handle the filtering within your `_read()`
        method, in which case you should set `manual_distributed_sharding` to `True`.

        See the notes below about how to do this.

    manual_multi_process_sharding : `bool`, optional (default=`False`)
        This is similar to the `manual_distributed_sharding` parameter, but applies to
        multi-process data loading. By default, when this reader is used by a multi-process
        data loader (i.e. a `DataLoader` with `num_workers > 1`), each worker will
        filter out all but a subset of the instances that are needed so that you
        don't end up with duplicates.

        However, there is really no benefit to using multiple workers in your `DataLoader`
        unless you implement handle the sharding within your `_read()` method, in which
        case you should set `manual_multi_process_sharding` to `True`, just as with
        `manual_distributed_sharding`.

        See the note below about how to do this.

    # Notes

    The default mechanism for filtering out `Instance`s in the distributed or multi-process
    `DataLoader` setting is not very efficient, since every worker would still need to
    process every single `Instance` in your dataset.

    This can be improved by manually handling the filtering / sharding within your `_read()`
    method.

    For example, if you were training using 2 GPUs and your `_read()` method reads a file
    line-by-line, creating one `Instance` for each line, you could just check the node
    rank within `_read()` and then throw away every other line starting at the line number
    corresponding to the node rank.

    The helper method `shard_iterable()` is there to make this easy for you.
    You can wrap this around any iterable object in your `_read()` method, and it will
    return an iterator that skips the right items based on the distributed training
    or multi-process loading context. This method can always be called regardless
    of whether or not you're actually using distributed training or multi-process loading.

    Remember though that when you handle the sharding manually within `_read()`, you need
    to let the `DatasetReader` know about this so that it doesn't do any additional
    filtering. Therefore you need to ensure that both `self.manual_distributed_sharding` and
    `self.manual_multi_process_sharding` are set to `True`.

    If you call the helper method `shard_iterable()` without setting these to `True`,
    you get an exception.

    """

    def __init__(
        self,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multi_process_sharding: bool = False,
    ) -> None:
        # Do some validation.
        if max_instances is not None and max_instances < 0:
            raise ValueError("If specified, max_instances should be a positive int")

        self.max_instances = max_instances
        self.manual_distributed_sharding = manual_distributed_sharding
        self.manual_multi_process_sharding = manual_multi_process_sharding
        self.__worker_info: Optional[WorkerInfo] = None

    def read(self, file_path: Union[PathLike, str]) -> Iterator[Instance]:
        """
        Returns an iterator of instances that can be read from the file path.
        """
        if not isinstance(file_path, str):
            file_path = str(file_path)

        for instance in self._multi_worker_islice(self._read(file_path)):
            yield instance

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable`.

        You are strongly encouraged to use a generator so that users can
        read a dataset in a lazy way, if they so choose.
        """
        raise NotImplementedError

    def text_to_instance(self, *inputs) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        `Instance`.  The primary intended use for this is with a
        :class:`~allennlp.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`_read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        `DatasetReader` to process new text lets us accomplish this, as we can just call
        `DatasetReader.text_to_instance` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The `Predictor` will
        have to make some assumptions about the kind of `DatasetReader` that it's using, in order
        to pass it the right information.
        """
        raise NotImplementedError

    def get_worker_info(self) -> Optional[WorkerInfo]:
        """
        Provides a [`WorkerInfo`](#WorkerInfo) object when the reader is being used within a
        worker of a multi-process `DataLoader`.

        If the reader is in the main process, this is just `None`.

        !!! NOTE
            This is different than distributed training. If the `DatasetReader`
            is being used within distributed training, `get_worker_info()` will only
            provide information on the `DataLoader` worker within its node.

        """
        return self.__worker_info

    def _set_worker_info(self, info: Optional[WorkerInfo]) -> None:
        """
        Should only be used internally.
        """
        self.__worker_info = info

    def shard_iterable(self, iterable: Iterable[_T]) -> Iterator[_T]:
        """
        Helper method that determines which items in an iterable object to skip based
        on the current node rank (for distributed training) and worker ID (for multi-process data loading).
        """
        if not self.manual_distributed_sharding or not self.manual_multi_process_sharding:
            raise ValueError(
                "self.shard_iterable() was called but self.manual_distributed_sharding and "
                "self.manual_multi_process_sharding was not set to True. Did you forget to call "
                "super().__init__(manual_distributed_sharding=True, manual_multi_process_sharding=True) "
                "in your constructor?"
            )

        sharded_slice: Iterator[_T] = iter(iterable)

        if util.is_distributed():
            sharded_slice = itertools.islice(
                sharded_slice, dist.get_rank(), None, dist.get_world_size()
            )

        if self.__worker_info is not None:
            sharded_slice = itertools.islice(
                sharded_slice, self.__worker_info.id, None, self.__worker_info.num_workers
            )

        return sharded_slice

    def _multi_worker_islice(self, iterable: Iterable[_T],) -> Iterator[_T]:
        """
        This is just like `shard_iterable` but is for internal use only.

        It has some additional logic to handle `max_instances` based on the distributed
        or multi-process context, and whether or not sharding is handled manually
        in the `_read()` method.
        """
        # This has some complicated logic because any given reader may or may not
        # implement manual multi-process and manual distributed sharding itself.
        # We have to handle all possibilities.

        sharded_slice: Iterator[_T] = iter(iterable)

        # We'll adjust max_instances as we go, depending on what sort of sharding is done.
        # At the end, we want to ensure the total number of instances collected across
        # all workers processes is equal to self.max_instances.
        max_instances = self.max_instances

        if util.is_distributed():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            if max_instances is not None:
                # Need to scale down max_instances because otherwise each node would read self.max_instances,
                # but we really want self.max_instances total across all nodes.
                if rank < (max_instances % world_size):
                    max_instances = max_instances // world_size + 1
                else:
                    max_instances = max_instances // world_size

            if not self.manual_distributed_sharding:
                sharded_slice = itertools.islice(sharded_slice, rank, None, world_size)

        if self.__worker_info is not None:
            if max_instances is not None:
                # Like in the distributed case above, we need to adjust max_instances.
                if self.__worker_info.id < (max_instances % self.__worker_info.num_workers):
                    max_instances = max_instances // self.__worker_info.num_workers + 1
                else:
                    max_instances = max_instances // self.__worker_info.num_workers

            if not self.manual_multi_process_sharding:
                warnings.warn(
                    "Using multi-process data loading without setting "
                    "DatasetReader.manual_multi_process_sharding to True.\n"
                    "Did you forget to set this?\n"
                    "If you're not handling the multi-process sharding logic within your "
                    "_read() method, there is probably no benefit to using more than one "
                    "worker.",
                    UserWarning,
                )
                sharded_slice = itertools.islice(
                    sharded_slice, self.__worker_info.id, None, self.__worker_info.num_workers
                )

        if max_instances is not None:
            sharded_slice = itertools.islice(sharded_slice, max_instances)

        return sharded_slice
