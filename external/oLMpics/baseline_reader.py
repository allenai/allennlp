# Exactly the same as the other dataset reader

from typing import Dict, List
import json
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

#from oLMpics.common.file_utils import cached_path
from allennlp.common.file_utils import cached_path
import gzip

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("base_esim_reader")
class BaselineEsimReader(DatasetReader):
    """
    Reads a file from the CommonsenseQA dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the choices.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the question and the choices.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_only_gold_examples: bool = False) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_only_gold_examples = use_only_gold_examples

    @overrides
    def _read(self, file_path: str):
        label_dict = {'A': 0, 'B': 1, 'C': 2}
        with gzip.open(cached_path(file_path), "rb") as f:
            for line_num, line in enumerate(f):
                # line = line.strip("\n")
                line = json.loads(line)
                if not line:
                    continue
                question = line['question']['stem']
                choices = [c['text'] for c in line['question']['choices']]
                label = label_dict[line['answerKey']] if 'answerKey' in line else None
                yield self.text_to_instance(question, choices, label=label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         choices: List[str],
                         label: int = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        question_tokens = self._tokenizer.tokenize(question)
        fields['premise'] = TextField(question_tokens, self._token_indexers)

        # This could be another way to get randomness
        for i, cho in enumerate(choices):
            choice_tokens = self._tokenizer.tokenize(cho)
            fields['hypothesis{}'.format(i)] = TextField(choice_tokens, self._token_indexers)

        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=True)
        return Instance(fields)

    # @classmethod
    # def from_params(cls, params: Params) -> 'BaselineEsimReader':
    #     tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
    #     token_indexers = TokenIndexer.from_params(params.pop('token_indexers', {}))
    #     params.assert_empty(cls.__name__)
    #     return cls(tokenizer=tokenizer,
    #                token_indexers=token_indexers)
