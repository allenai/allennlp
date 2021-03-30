from dataclasses import dataclass
import itertools
from os import PathLike
from typing import Iterable, Iterator, Optional, Union, TypeVar, Dict, List
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


@dataclass
class DistributedInfo:
    """
    Contains information about the global process rank and total world size when the reader is being
    used within distributed training.

    From a `DatasetReader` this can be accessed with the [`get_distributed_info()`](#get_distributed_info) method.
    """

    world_size: int
    """
    The total number of processes in the distributed group.
    """

    global_rank: int
    """
    The 0-indexed ID of the current process within the distributed group.
    This will be between 0 and `world_size - 1`, inclusive.
    """


_T = TypeVar("_T")

PathOrStr = Union[PathLike, str]
DatasetReaderInput = Union[PathOrStr, List[PathOrStr], Dict[str, PathOrStr]]


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
        trainer process only receives a subset of the data. It does this by reading the whole
        dataset in each worker, but filtering out the instances that are not needed.

        While this ensures that each worker will recieve unique instances, it's not a very efficient
        way to do so since each worker still needs to process every single instance.

        A better way to handle this is to manually handle the filtering within your `_read()`
        method, in which case you should set `manual_distributed_sharding` to `True` so that
        the base class knows that you handling the filtering.

        See the section below about how to do this.

    manual_multiprocess_sharding : `bool`, optional (default=`False`)
        This is similar to the `manual_distributed_sharding` parameter, but applies to
        multi-process data loading. By default, when this reader is used by a multi-process
        data loader (i.e. a `DataLoader` with `num_workers > 1`), each worker will
        filter out all but a subset of the instances that are needed so that you
        don't end up with duplicates.

        However, there is really no benefit to using multiple workers in your `DataLoader`
        unless you implement the sharding within your `_read()` method, in which
        case you should set `manual_multiprocess_sharding` to `True`, just as with
        `manual_distributed_sharding`.

        See the section below about how to do this.

    serialization_dir: `str`, optional (default=`None`)
        The directory in which the training output is saved to, or the directory the model is loaded from.

        !!! Note
            This is typically not given an entry in a configuration file. It will be set automatically
            when using the built-in `allennp` commands.

    # Using your reader with multi-process or distributed data loading

    There are two things you may need to update in your `DatasetReader` in order for
    it to be efficient in the multi-process or distributed data loading context.

    1. The `_read()` method should handle filtering out all but the instances that
        each particular worker should generate.

        This is important because the default mechanism for filtering out `Instance`s in
        the distributed or multi-process `DataLoader` setting is not very efficient, since every
        worker would still need to process every single `Instance` in your dataset.

        But by manually handling the filtering / sharding within your `_read()` method, each
        worker only needs to perform a subset of the work required to create instances.

        For example, if you were training using 2 GPUs and your `_read()` method reads a file
        line-by-line, creating one `Instance` for each line, you could just check the node
        rank within `_read()` and then throw away every other line starting at the line number
        corresponding to the node rank.

        The helper method [`shard_iterable()`](#shard_iterable) is there to make this easy for you.
        You can wrap this around any iterable object in your `_read()` method, and it will
        return an iterator that skips the right items based on the distributed training
        or multi-process loading context. This method can always be called regardless
        of whether or not you're actually using distributed training or multi-process loading.

        Remember though that when you handle the sharding manually within `_read()`, you need
        to let the `DatasetReader` know about this so that it doesn't do any additional
        filtering. Therefore you need to ensure that both `self.manual_distributed_sharding` and
        `self.manual_multiprocess_sharding` are set to `True`.

        If you call the helper method `shard_iterable()` without setting these to `True`,
        you'll get an exception.

    2. If the instances generated by `_read()` contain `TextField`s, those `TextField`s
        should not have any token indexers assigned. The token indexers need to be applied
        in the [`apply_token_indexers()`](#apply_token_indexers) method instead.

        This is highly recommended because if the instances generated by your `_read()` method
        have token indexers attached, those indexers will be duplicated when they are sent across
        processes. If your token indexers contain large objects (such as `PretrainedTransformerTokenIndexer`s)
        this could take up a massive amount of memory.

    """

    def __init__(
        self,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multiprocess_sharding: bool = False,
        serialization_dir: Optional[str] = None,
    ) -> None:
        # Do some validation.
        if max_instances is not None and max_instances < 0:
            raise ValueError("If specified, max_instances should be a positive int")

        self.max_instances = max_instances
        self.manual_distributed_sharding = manual_distributed_sharding
        self.manual_multiprocess_sharding = manual_multiprocess_sharding
        self.serialization_dir = serialization_dir
        self._worker_info: Optional[WorkerInfo] = None
        self._distributed_info: Optional[DistributedInfo] = None
        # If we're actually in the main process, we can find the info using torch utils.
        if util.is_distributed():
            self._distributed_info = DistributedInfo(dist.get_world_size(), dist.get_rank())

    def read(self, file_path: DatasetReaderInput) -> Iterator[Instance]:
        """
        Returns an iterator of instances that can be read from the file path.
        """
        for instance in self._multi_worker_islice(self._read(file_path)):  # type: ignore
            if self._worker_info is None:
                # If not running in a subprocess, it's safe to apply the token_indexers right away.
                self.apply_token_indexers(instance)
            yield instance

    def _read(self, file_path) -> Iterable[Instance]:
        """
        Reads the instances from the given `file_path` and returns them as an
        `Iterable`.

        You are strongly encouraged to use a generator so that users can
        read a dataset in a lazy way, if they so choose.
        """
        # NOTE: `file_path` is left untyped here on purpose.
        # Technically the type should be `DatasetReaderInput`, but many subclass
        # implementations of `DatasetReader` define their `_read()` method to take a more
        # specific type, such as just `str`. But that would be a type error
        # according to mypy: https://mypy.readthedocs.io/en/stable/common_issues.html#incompatible-overrides
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

    def apply_token_indexers(self, instance: Instance) -> None:
        """
        If `Instance`s created by this reader contain `TextField`s without `token_indexers`,
        this method can be overriden to set the `token_indexers` of those fields.

        E.g. if you have you have `"source"` `TextField`, you could implement this method like this:

        ```python
        def apply_token_indexers(self, instance: Instance) -> None:
            instance["source"].token_indexers = self._token_indexers
        ```

        If your `TextField`s are wrapped in a `ListField`, you can access them via `field_list`.
        E.g. if you had a `"source"` field of `ListField[TextField]` objects, you could:

        ```python
        for text_field in instance["source"].field_list:
            text_field.token_indexers = self._token_indexers
        ```
        """
        pass

    def get_worker_info(self) -> Optional[WorkerInfo]:
        """
        Provides a [`WorkerInfo`](#WorkerInfo) object when the reader is being used within a
        worker of a multi-process `DataLoader`.

        If the reader is in the main process, this is just `None`.

        !!! NOTE
            This is different than distributed training. If the `DatasetReader`
            is being used within distributed training, `get_worker_info()` will only
            provide information on the `DataLoader` worker within its node.

            Use [`get_distributed_info`](#get_distributed_info) to get information on distributed
            training context.

        """
        return self._worker_info

    def get_distributed_info(self) -> Optional[DistributedInfo]:
        """
        Provides a [`DistributedInfo`](#DistributedInfo) object when the reader is being
        used within distributed training.

        If not in distributed training, this is just `None`.
        """
        return self._distributed_info

    def _set_worker_info(self, info: Optional[WorkerInfo]) -> None:
        """
        Should only be used internally.
        """
        self._worker_info = info

    def _set_distributed_info(self, info: Optional[DistributedInfo]) -> None:
        """
        Should only be used internally.
        """
        self._distributed_info = info

    def shard_iterable(self, iterable: Iterable[_T]) -> Iterator[_T]:
        """
        Helper method that determines which items in an iterable object to skip based
        on the current node rank (for distributed training) and worker ID (for multi-process data loading).
        """
        if not self.manual_distributed_sharding or not self.manual_multiprocess_sharding:
            raise ValueError(
                "self.shard_iterable() was called but self.manual_distributed_sharding and "
                "self.manual_multiprocess_sharding was not set to True. Did you forget to call "
                "super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True) "
                "in your constructor?"
            )

        sharded_slice: Iterator[_T] = iter(iterable)

        if util.is_distributed():
            sharded_slice = itertools.islice(
                sharded_slice, dist.get_rank(), None, dist.get_world_size()
            )

        if self._worker_info is not None:
            sharded_slice = itertools.islice(
                sharded_slice, self._worker_info.id, None, self._worker_info.num_workers
            )

        # We don't know for sure how many instances we have to produce.
        # _multi_worker_islice() figures that out. But we know for sure
        # it won't be more than max_instances.
        if self.max_instances is not None:
            sharded_slice = itertools.islice(sharded_slice, self.max_instances)

        return sharded_slice

    def _multi_worker_islice(
        self,
        iterable: Iterable[_T],
    ) -> Iterator[_T]:
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

        if self._distributed_info is not None:
            if max_instances is not None:
                # Need to scale down max_instances because otherwise each node would read self.max_instances,
                # but we really want self.max_instances total across all nodes.
                if self._distributed_info.global_rank < (
                    max_instances % self._distributed_info.world_size
                ):
                    max_instances = max_instances // self._distributed_info.world_size + 1
                else:
                    max_instances = max_instances // self._distributed_info.world_size

            if not self.manual_distributed_sharding:
                sharded_slice = itertools.islice(
                    sharded_slice,
                    self._distributed_info.global_rank,
                    None,
                    self._distributed_info.world_size,
                )

        if self._worker_info is not None:
            if max_instances is not None:
                # Like in the distributed case above, we need to adjust max_instances.
                if self._worker_info.id < (max_instances % self._worker_info.num_workers):
                    max_instances = max_instances // self._worker_info.num_workers + 1
                else:
                    max_instances = max_instances // self._worker_info.num_workers

            if not self.manual_multiprocess_sharding:
                warnings.warn(
                    "Using multi-process data loading without setting "
                    "DatasetReader.manual_multiprocess_sharding to True.\n"
                    "Did you forget to set this?\n"
                    "If you're not handling the multi-process sharding logic within your "
                    "_read() method, there is probably no benefit to using more than one "
                    "worker.",
                    UserWarning,
                )
                sharded_slice = itertools.islice(
                    sharded_slice, self._worker_info.id, None, self._worker_info.num_workers
                )

        if max_instances is not None:
            sharded_slice = itertools.islice(sharded_slice, max_instances)

        return sharded_slice
