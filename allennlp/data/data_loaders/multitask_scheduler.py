from typing import Any, Dict, Iterable, Tuple

from allennlp.common.registrable import Registrable
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
