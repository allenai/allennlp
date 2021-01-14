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


@MultiTaskEpochSampler.register("uniform")
class UniformSampler(MultiTaskEpochSampler):
    """
    Returns a uniform distribution over datasets at every epoch.

    Registered as a `MultiTaskEpochSampler` with name "uniform".
    """

    def get_task_proportions(self, data_loaders: Mapping[str, DataLoader]) -> Dict[str, float]:
        return {key: 1 / len(data_loaders) for key in data_loaders}


@MultiTaskEpochSampler.register("weighted")
class WeightedSampler(MultiTaskEpochSampler):
    """
    Returns a weighted distribution over datasets at every epoch, where every
    task has a weight.

    Registered as a `MultiTaskEpochSampler` with name "weighted".
    """

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def get_task_proportions(self, data_loaders: Mapping[str, DataLoader]) -> Dict[str, float]:
        total = sum(self.weights[task] for task in data_loaders.keys())
        return {task: self.weights[task] / total for task in data_loaders.keys()}


@MultiTaskEpochSampler.register("proportional")
class ProportionalSampler(MultiTaskEpochSampler):
    """
    Samples from every dataset according to its size.  This will have essentially the same effect as
    using all of the data at every epoch, but it lets you control for number of instances per epoch,
    if you want to do that.  This requires that all data loaders have a `__len__` (which means no
    lazy loading).  If you need this functionality with lazy loading, implement your own sampler
    that takes dataset sizes as a constructor parameter.

    Registered as a `MultiTaskEpochSampler` with name "proportional".
    """

    def get_task_proportions(self, data_loaders: Mapping[str, DataLoader]) -> Dict[str, float]:
        try:
            sizes = {key: len(loader) for key, loader in data_loaders.items()}
        except TypeError:
            raise ValueError("ProportionalSampler got passed a data loader without a length")
        total_size = sum(sizes.values())
        return {key: size / total_size for key, size in sizes.items()}
