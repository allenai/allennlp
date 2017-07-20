from allennlp.data.dataset import Dataset
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
        from allennlp.experiments.registry import Registry
        choice = params.pop_choice('type', Registry.list_dataset_readers())
        return Registry.get_dataset_reader(choice).from_params(params)
