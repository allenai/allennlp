from typing import Dict
import json
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("snli")
class SnliReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """
    def __init__(self,
                 tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(tokenizer=tokenizer, token_indexers=token_indexers)

    @overrides
    def read(self, file_path: str):
        instances = []
        with open(file_path, 'r') as snli_file:
            logger.info("Reading SNLI instances from jsonl dataset at: %s", file_path)
            for line in tqdm.tqdm(snli_file):
                example = json.loads(line)

                label = example["gold_label"]
                label_field = LabelField(label)

                premise = example["sentence1"]
                premise_field = TextField(self._tokenizer.tokenize(premise), self._token_indexers)
                hypothesis = example["sentence2"]
                hypothesis_field = TextField(self._tokenizer.tokenize(hypothesis), self._token_indexers)
                instances.append(Instance({'label': label_field,
                                           'premise': premise_field,
                                           'hypothesis': hypothesis_field}))
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @classmethod
    def from_params(cls, params: Params) -> 'SnliReader':
        """
        Parameters
        ----------
        filename : ``str``
        tokenizer : ``Params``, optional
        token_indexers: ``List[Params]``, optional
        """
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = {}
        token_indexer_params = params.pop('token_indexers', Params({}))
        for name, indexer_params in token_indexer_params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)
        # The default parameters are contained within the class,
        # so if no parameters are given we must pass None.
        if token_indexers == {}:
            token_indexers = None
        params.assert_empty(cls.__name__)
        return SnliReader(tokenizer=tokenizer,
                          token_indexers=token_indexers)
