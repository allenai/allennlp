from allennlp.data.dataset import Dataset
from allennlp.common import Params


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

    @classmethod
    def from_params(cls, params: Params):
        from allennlp.experiments.registry import Registry
        choice = params.pop_choice('type', Registry.get_dataset_readers())
        return Registry.get_dataset_reader(choice).from_params(params)
