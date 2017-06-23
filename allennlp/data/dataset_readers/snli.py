from typing import List
import json

from overrides import overrides

from . import DatasetReader
from .. import Dataset
from .. import Instance
from ...common import Params
from ..fields import TextField, LabelField
from ..token_indexers import TokenIndexer, SingleIdTokenIndexer
from ..tokenizers import Tokenizer, WordTokenizer


class SnliReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------
    snli_filename : ``str``
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``List[TokenIndexer]``, optional (default=``[SingleIdTokenIndexer()]``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """
    def __init__(self,
                 snli_filename: str,
                 tokenizer: Tokenizer=WordTokenizer(),
                 token_indexers: List[TokenIndexer]=None):
        self._snli_filename = snli_filename
        self._tokenizer = tokenizer
        if token_indexers is None:
            token_indexers = [SingleIdTokenIndexer()]
        self._token_indexers = token_indexers

    @overrides
    def read(self):
        instances = []
        with open(self._snli_filename, 'r') as snli_file:
            for line in snli_file:
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
        return Dataset(instances)

    @classmethod
    def from_params(cls, params: Params):
        """
        Parameters
        ----------
        filename : ``str``
        tokenizer : ``Params``, optional
        token_indexers: ``List[Params]``, optional
        """
        filename = params.pop('filename')
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = [TokenIndexer.from_params(p)
                          for p in params.pop('token_indexers', [Params({})])]
        params.assert_empty(cls.__name__)
        return SnliReader(snli_filename=filename,
                          tokenizer=tokenizer,
                          token_indexers=token_indexers)
