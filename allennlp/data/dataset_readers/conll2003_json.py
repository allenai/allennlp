from typing import Dict, List
import logging
import json

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("conll2003_json")
class Conll2003JsonDatasetReader(DatasetReader):
    """
    Reads instances from a JsonL file where each line is a Json blob representing a single sentence.

    Parameters
    ----------

    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            for line in tqdm.tqdm(data_file):
                data = json.loads(line)
                tokens = [Token(token['token']) for token in data]
                labels = [token['labels']['conll2003'] for token in data]

                sequence = TextField(tokens, self._token_indexers)
                instance = Instance({
                        'tokens': sequence,
                        'tags': SequenceLabelField(labels, sequence)
                })

                instances.append(instance)

        return Dataset(instances)

    def text_to_instance(self, tokens: List[Token]) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        return Instance({'tokens': TextField(tokens, token_indexers=self._token_indexers)})

    @classmethod
    def from_params(cls, params: Params) -> 'Conll2003JsonDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return Conll2003JsonDatasetReader(token_indexers=token_indexers)
