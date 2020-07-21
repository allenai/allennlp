from dataclasses import dataclass
import itertools
from typing import Iterable, Iterator, Optional, Any, Callable, Union
import logging
from pathlib import Path
import warnings

import torch.distributed as dist

from allennlp.data.instance import Instance
from allennlp.common import util
from allennlp.common.registrable import Registrable


logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    num_workers: int
    id: int


class DatasetReader(Registrable):
    """
    A `DatasetReader` knows how to turn a file containing a dataset into a collection
    of `Instances`.  To implement your own, just override the `_read(file_path)` method
    to return an `Iterable` of the instances. This could be a list containing the instances
    or a lazy generator that returns them one at a time.

    All parameters necessary to `_read` the data apart from the filepath should be passed
    to the constructor of the `DatasetReader`.

    # Parameters

    max_instances : `int`, optional (default=`None`)
        If given, will stop reading after this many instances. This is a useful setting for debugging.
        Setting this disables caching.

    manual_distributed_sharding: `bool`, optional (default=`False`)
        By default, when used in a distributed setting, `DatasetReader` makes sure that each
        worker process only receives a subset of the data. It does this by reading the whole
        dataset in each worker, but filtering out the instances that are not needed. If you
        can implement a faster mechanism that only reads part of the data, set this to True,
        and do the sharding yourself.

    manual_multi_process_sharding : `bool`, optional (default=`False`)
        This is similar to the `manual_distributed_sharding` parameter, but applies to
        multi-process data loading. By default, when this reader is used by a multi-process
        data loader (i.e. a `DataLoader` with `num_workers > 1`), each worker will
        filter out all but a subset of the instances that are needed so that you
        don't end up with duplicates.

        !!! NOTE
            **There is really no benefit of using a multi-process
            `DataLoader` unless you can specifically implement a faster sharding mechanism
            within `_read()`**. In that case you should set `manual_multi_process_sharding`
            to `True`.

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
        self._worker_info: Optional[WorkerInfo] = None

    def read(self, file_path: Union[Path, str]) -> Iterator[Instance]:
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
        `Iterable` (which could be a list or could be a generator).

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

    @property
    def worker_info(self) -> Optional[WorkerInfo]:
        """
        Provides a `WorkerInfo` object when the reader is being used within a worker process.
        If the reader is in the main process, this is just `None`.
        """
        return self._worker_info

    @worker_info.setter
    def worker_info(self, info: Optional[WorkerInfo]) -> None:
        self._worker_info = info

    def _multi_worker_islice(
        self,
        iterable: Iterable[Any],
        transform: Optional[Callable[[Any], Instance]] = None,
        ensure_lazy: bool = False,
    ) -> Iterable[Instance]:
        """
        Helper method that determines which raw instances to skip based on the current
        node rank (for distributed training) and worker ID (for multi-process data loading).

        # Parameters

        iterable : `Iterable[Any]`
            An iterable that yields raw data that can be transformed into `Instance`s
            through the `transform` function.
        transform : `Optional[Callable[[Any], Instance]]`, optional (default = `None`)
            An optional function that will be applied to the raw data generated
            by `iterable` to create `Instance`s. This is used, e.g., when reading
            cached data.
        ensure_lazy : `bool`, optional (default = `False`)
            If `True`, a `ConfigurationError` error will be raised if `iterable`
            is a list instead of a lazy generator type.

        # Returns

        `Iterable[Instance]`
        """
        # This has some complicated logic because any given reader may or may not
        # implement manual multi-process and manual distributed sharding itself.
        # We have to handle all possibilities.

        sharded_slice: Iterable[Any] = iterable

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
                start_index = rank
                step_size = world_size
                sharded_slice = itertools.islice(sharded_slice, rank, None, world_size)

        if self.worker_info is not None:
            if max_instances is not None:
                # Like in the distributed case above, we need to adjust max_instances.
                if self.worker_info.id < (max_instances % self.worker_info.num_workers):
                    max_instances = max_instances // self.worker_info.num_workers + 1
                else:
                    max_instances = max_instances // self.worker_info.num_workers

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
                    sharded_slice, self.worker_info.id, None, self.worker_info.num_workers
                )

        if max_instances is not None:
            sharded_slice = itertools.islice(sharded_slice, max_instances)

        if transform is not None:
            sharded_slice = (transform(x) for x in sharded_slice)

        return sharded_slice
