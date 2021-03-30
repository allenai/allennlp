from collections import defaultdict
from typing import Any, Dict, Iterable, Union, List, Mapping

import more_itertools

from allennlp.common.registrable import Registrable
from allennlp.data.instance import Instance


class MultiTaskScheduler(Registrable):
    """
    A class that determines how to order instances within an epoch.
    This is used by the `MultiTaskDataLoader`. The main operation performed by this class is to
    take a dictionary of instance iterators, one for each dataset, and combine them into an
    iterator of batches, based on some scheduling algorithm (such as round robin, randomly choosing
    between available datasets, etc.). To control this behavior as training progresses, there is an
    `update_from_epoch_metrics` method available, which should be called from a `Callback` during
    training.  Not all `MultiTaskSchedulers` will implement this method.
    """

    def batch_instances(
        self, epoch_instances: Dict[str, Iterable[Instance]]
    ) -> Iterable[List[Instance]]:
        """
        Given a dictionary of `Iterable[Instance]` for each dataset, combines them into an
        `Iterable` of batches of instances.
        """
        raise NotImplementedError

    def update_from_epoch_metrics(self, epoch_metrics: Dict[str, Any]) -> None:
        """
        In case you want to set the behavior of the scheduler based on current epoch metrics, you
        can do that by calling this method from a `Callback`.  If your scheduling technique does not
        depend on epoch metrics, you do not need to implement this method.
        """
        raise NotImplementedError

    def count_batches(self, dataset_counts: Dict[str, int]) -> int:
        """
        Given the number of instances per dataset, this returns the total number of batches
        the scheduler will return.
        """
        raise NotImplementedError

    default_implementation = "homogeneous_roundrobin"


def _chunked_iterator(i: Iterable, chunk_size: int, drop_last: bool):
    chunks = more_itertools.chunked(i, chunk_size)
    if drop_last:
        return (chunk for chunk in chunks if len(chunk) == chunk_size)
    else:
        return chunks


@MultiTaskScheduler.register("roundrobin")
class RoundRobinScheduler(MultiTaskScheduler):
    """
    Orders instances in a round-robin fashion, where we take one instance from every dataset in
    turn. When one dataset runs out, we continue iterating round-robin through the rest.

    Registered as a `MultiTaskScheduler` with name "roundrobin".
    """

    def __init__(self, batch_size: int, drop_last: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.drop_last = drop_last

    def batch_instances(
        self, epoch_instances: Dict[str, Iterable[Instance]]
    ) -> Iterable[List[Instance]]:
        return _chunked_iterator(
            more_itertools.roundrobin(*epoch_instances.values()), self.batch_size, self.drop_last
        )

    def count_batches(self, dataset_counts: Dict[str, int]) -> int:
        instance_count = sum(dataset_counts.values())
        if self.drop_last or instance_count % self.batch_size == 0:
            return instance_count // self.batch_size
        else:
            return 1 + (instance_count // self.batch_size)


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
        names used elsewhere in the multi-task code.
    """

    def __init__(self, batch_size: Union[int, Dict[str, int]], drop_last: bool = False):
        self.batch_size: Mapping[str, int]
        if isinstance(batch_size, int):
            self.batch_size = defaultdict(lambda: batch_size)  # type: ignore
        else:
            self.batch_size = batch_size
        self.drop_last = drop_last

    def batch_instances(
        self, epoch_instances: Dict[str, Iterable[Instance]]
    ) -> Iterable[List[Instance]]:
        chunked_iterators = [
            _chunked_iterator(iterator, self.batch_size[dataset], self.drop_last)
            for dataset, iterator in epoch_instances.items()
        ]
        return more_itertools.roundrobin(*chunked_iterators)

    def count_batches(self, dataset_counts: Dict[str, int]) -> int:
        result = 0
        for dataset, count in dataset_counts.items():
            batch_size = self.batch_size[dataset]
            result += count // batch_size
            if not self.drop_last and count % batch_size != 0:
                result += 1
        return result
