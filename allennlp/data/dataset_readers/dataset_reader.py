from typing import Dict

from allennlp.data.dataset import Dataset
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common import Params
from allennlp.common.registrable import Registrable


class DatasetReader(Registrable):
    """
    A ``DatasetReader`` reads data from some location and constructs a :class:`Dataset`.  All
    parameters necessary to read the data apart from the filepath should be passed to the
    constructor of the ``DatasetReader``.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None) -> None:
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def read(self, file_path: str) -> Dataset:
        """
        Actually reads some data from the `file_path` and returns a :class:`Dataset`.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'DatasetReader':
        """
        Static method that constructs the dataset reader described by ``params``.
        """
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)
