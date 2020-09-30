from typing import Any, Dict, Mapping

from allennlp.common.registrable import Registrable
from allennlp.data.data_loaders.data_loader import DataLoader


class MultiTaskEpochSampler(Registrable):
    """
    A class that determines with what proportion each dataset should be sampled for a given epoch.
    This is used by the `MultiTaskDataLoader`.  The main output of this class is the task proportion
    dictionary returned by `get_task_proportions`, which specifies what percentage of the instances
    for the current epoch should come from each dataset.  To control this behavior as training
    progresses, there is an `update_from_epoch_metrics` method, which should be called from a
    `Callback` during training.
    """

    def get_task_proportions(self, data_loaders: Mapping[str, DataLoader]) -> Dict[str, float]:
        """
        Given a dictionary of `DataLoaders` for each dataset, returns what percentage of the
        instances for the current epoch of training should come from each dataset.  The input
        dictionary could be used to determine how many datasets there are (e.g., for uniform
        sampling) or how big each dataset is (e.g., for sampling based on size), or it could be
        ignored entirely.
        """
        raise NotImplementedError

    def update_from_epoch_metrics(self, epoch_metrics: Dict[str, Any]) -> None:
        """
        Some implementations of EpochSamplers change their behavior based on current epoch metrics.
        This method is meant to be called from a `Callback`, to let the sampler update its sampling
        proportions.  If your sampling technique does not depend on epoch metrics, you do not need
        to implement this method.
        """
        raise NotImplementedError
