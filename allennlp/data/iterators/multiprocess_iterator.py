import warnings

from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.transform_iterator import TransformIterator
from allennlp.data import transforms


@DataIterator.register("multiprocess")
class MultiprocessIterator(DataIterator):
    """
    Wraps another ```DataIterator``` and uses it to generate tensor dicts
    using multiple processes.

    Parameters
    ----------
    base_iterator : ``DataIterator``
        The ``DataIterator`` for generating tensor dicts. It will be shared among
        processes, so it should not be stateful in any way.
    num_workers : ``int``, optional (default = 1)
        The number of processes used for generating tensor dicts.
    output_queue_size: ``int``, optional (default = 1000)
        The size of the output queue on which tensor dicts are placed to be consumed.
        You might need to increase this if you're generating tensor dicts too quickly.
    """

    def __new__(
        cls, base_iterator: TransformIterator, num_workers: int = 1, output_queue_size: int = 1000
    ):

        warnings.warn(
            "The MultiprocessIterator is depreciated. "
            "Instead, you can simply pass the num_workers argument to your base iterator.",
            FutureWarning,
        )
        base_iterator._num_workers = num_workers
        base_iterator.transforms.append(transforms.Fork())

        return base_iterator

    def __init__(
        self, base_iterator: TransformIterator, num_workers: int = 1, output_queue_size: int = 1000
    ) -> None:
        pass
