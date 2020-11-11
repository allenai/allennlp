from collections import defaultdict
import itertools
from typing import Any, Dict, Iterable, Tuple, Union

import more_itertools

from allennlp.common.registrable import Registrable
from allennlp.common import util
from allennlp.data.instance import Instance


class MultiTaskScheduler(Registrable):
    """
    A class that determines how to order instances within an epoch.
    This is used by the `MultiTaskDataLoader`.  The main operation performed by this class is to
    take a dictionary of instance iterators, one for each dataset, and combine them into a single
    iterator, based on some scheduling algorithm (such as round robin, randomly choosing between
    available datasets, etc.).  To control this behavior as training progresses, there is an
    `update_from_epoch_metrics` method available, which should be called from a `Callback` during
    training.  Not all `MultiTaskSchedulers` will implement this method.
    """

    def order_epoch_instances(
        self, epoch_instances: Dict[str, Iterable[Instance]]
    ) -> Iterable[Tuple[str, Instance]]:
        """
        Given a dictionary of `Iterable[Instance]` for each dataset, combines them into a single
        `Iterable`, where the values returned by that iterator are (dataset, instance) tuples.
        """
        raise NotImplementedError

    def update_from_epoch_metrics(self, epoch_metrics: Dict[str, Any]) -> None:
        """
        In case you want to set the behavior of the scheduler based on current epoch metrics, you
        can do that by calling this method from a `Callback`.  If your scheduling technique does not
        depend on epoch metrics, you do not need to implement this method.
        """
        raise NotImplementedError


@MultiTaskScheduler.register("roundrobin")
class RoundRobinScheduler(MultiTaskScheduler):
    """
    Orders instances in a round-robin fashion, where we take one instance from every dataset in
    turn.  When one dataset runs out, we continue iterating round-robin through the rest.

    Registered as a `MultiTaskScheduler` with name "roundrobin".
    """

    def order_epoch_instances(
        self, epoch_instances: Dict[str, Iterable[Instance]]
    ) -> Iterable[Tuple[str, Instance]]:
        iterators = [
            zip(itertools.cycle([dataset]), iterator)
            for dataset, iterator in epoch_instances.items()
        ]
        return more_itertools.roundrobin(*iterators)


@MultiTaskScheduler.register("homogeneous_roundrobin")
class HomogeneousRoundRobinScheduler(MultiTaskScheduler):
    """
    Orders instances in a round-robin fashion, but grouped into batches composed entirely of
    instances from one dataset.  We'll return one batch from one dataset, then another batch from a
    different dataset, etc.  This is currently necessary in AllenNLP if your instances have
    different fields for different datasets, as we can't currently combine instances with different
    fields.

    When one dataset runs out, we continue iterating round-robin through the rest.

    If you want more fine-grained control over which datasets can be combined, it should be
    relatively straightforward to write your own scheduler, following this logic, which allows some
    datasets to be combined and others not.

    Registered as a `MultiTaskScheduler` with name "homogeneous_roundrobin".

    # Parameters

    batch_size: `Union[int, Dict[str, int]]`
        Determines how many instances to group together in each dataset.  If this is an `int`, the
        same value is used for all datasets; otherwise, the keys must correspond to the dataset
        names used elsewhere in the multi-task code.  Note also that this needs to match the batch
        size set in the `MultiTaskDataLoader`; because of how the ordering works, we will actually
        unroll the batching that we create here, so that the `MultiTaskDataLoader` can re-batch them
        (this is because not all ordering methods perform batching, so we do it in the data loader
        itself).
    """

    def __init__(self, batch_size: Union[int, Dict[str, int]]):
        if isinstance(batch_size, int):
            batch_size = defaultdict(lambda: batch_size)  # type: ignore
        self.batch_size = batch_size

    def order_epoch_instances(
        self, epoch_instances: Dict[str, Iterable[Instance]]
    ) -> Iterable[Tuple[str, Instance]]:
        grouped_iterators = [
            util.lazy_groups_of(zip(itertools.cycle([dataset]), iterator), self.batch_size[dataset])
            for dataset, iterator in epoch_instances.items()
        ]
        batch_iterator = more_itertools.roundrobin(*grouped_iterators)
        for batch in batch_iterator:
            yield from batch
