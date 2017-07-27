from typing import List, Dict, TypeVar, Type, Generic  # pylint: disable=unused-import

from allennlp.data.dataset import Dataset
from allennlp.common import Params
from allennlp.common.registrable import Registrable


class DatasetReader(Registrable):
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
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)
