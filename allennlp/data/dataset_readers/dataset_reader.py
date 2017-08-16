from typing import Dict, Optional

from allennlp.data.dataset import Dataset
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common import Params
from allennlp.common.registrable import Registrable


class DatasetReader(Registrable):
    """
    A ``DatasetReader`` reads data from some location and constructs a :class:`Dataset`.  All
    parameters necessary to read the data apart from the filepath should be passed to the
    constructor of the ``DatasetReader``.
    """
    def __init__(self):
        self._tokenizer = None      # type: Optional[Tokenizer]
        self._token_indexers = {}   # type: Dict[str, TokenIndexer]

    def read(self, file_path: str) -> Dataset:
        """
        Actually reads some data from the `file_path` and returns a :class:`Dataset`.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'DatasetReader':
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)
