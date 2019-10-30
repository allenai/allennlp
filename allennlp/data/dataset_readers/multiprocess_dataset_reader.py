import warnings

from allennlp.data.dataset_readers.dataset_reader import DatasetReader


@DatasetReader.register("multiprocess")
class MultiprocessDatasetReader(DatasetReader):
    """
    Wraps another dataset reader and uses it to read from multiple input files
    using multiple processes. Note that in this case the ``file_path`` passed to ``read()``
    should be a glob, and that the dataset reader will return instances from all files
    matching the glob.

    The order the files are processed in is a function of Numpy's random state
    up to non-determinism caused by using multiple worker processes. This can
    be avoided by setting ``num_workers`` to 1.

    Parameters
    ----------
    base_reader : ``DatasetReader``
        Each process will use this dataset reader to read zero or more files.
    num_workers : ``int``
        How many data-reading processes to run simultaneously.
    epochs_per_read : ``int``, (optional, default=1)
        Normally a call to ``DatasetReader.read()`` returns a single epoch worth of instances,
        and your ``DataIterator`` handles iteration over multiple epochs. However, in the
        multiple-process case, it's possible that you'd want finished workers to continue on to the
        next epoch even while others are still finishing the previous epoch. Passing in a value
        larger than 1 allows that to happen.
    output_queue_size: ``int``, (optional, default=1000)
        The size of the queue on which read instances are placed to be yielded.
        You might need to increase this if you're generating instances too quickly.
    """

    def __new__(
        cls,
        base_reader: DatasetReader,
        num_workers: int,
        epochs_per_read: int = 1,
        output_queue_size: int = 1000,
    ):

        warnings.warn(
            "The MultiProcessDataIterator is depreciated - constructing this "
            "class returns the base reader. Instead, pass num_workers to the iterator "
            "to achieve multi-process data loading.",
            FutureWarning,
        )

        return base_reader

    def __init__(
        self,
        base_reader: DatasetReader,
        num_workers: int,
        epochs_per_read: int = 1,
        output_queue_size: int = 1000,
    ):
        pass
