from allennlp.data import Dataset
from allennlp.common import Params


class DatasetReader:
    """
    A ``DatasetReader`` reads data from some location and constructs a :class:`Dataset`.  All
    parameters necessary to read the data apart from the filepath should be passed to the
    constructor of the ``DatasetReader``.
    """
    def read(self, file_path: str) -> Dataset:
        """
        Actually reads some data from the `file_path` and returns a :class:`Dataset`.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params):
        from allennlp.data.dataset_readers import dataset_readers
        choice = params.pop_choice('type', list(dataset_readers.keys()))
        return dataset_readers[choice].from_params(params)
