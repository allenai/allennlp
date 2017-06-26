from .. import Dataset
from ...common import Params


class DatasetReader:
    """
    A ``DatasetReader`` reads data from some location and constructs a :class:`Dataset`.  All
    parameters necessary to read the data, including the file path(s), should be passed to the
    constructor of the ``DatasetReader``.
    """
    def read(self) -> Dataset:
        """
        Actually reads some data and returns a :class:`Dataset`.
        """
        raise NotImplementedError

    @staticmethod
    def from_params(params: Params):
        from . import dataset_readers
        choice = params.pop_choice('type', list(dataset_readers.keys()))
        return dataset_readers[choice].from_params(params)
