# Exactly the same as the other dataset reader

from typing import Dict, List
import json
import logging
import random

from overrides import overrides

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

#from oLMpics.common.file_utils import cached_path
from allennlp.common.file_utils import cached_path
import gzip

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("base_mlm_reader")
class BaselineMLMReader(DatasetReader):
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
                 sample: int = -1,
                 use_only_gold_examples: bool = False) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_only_gold_examples = use_only_gold_examples
        self._sample = sample

    @overrides
    def _read(self, file_path: str):
        label_dict = {'A': 0, 'B': 1, 'C': 2, 'D':3 ,'E':4}
        examples = []
        with gzip.open(cached_path(file_path), "rb") as f:
            for line_num, line in enumerate(f):
                # line = line.strip("\n")
                line = json.loads(line)
                if not line:
                    continue
                examples.append(line)
        if self._sample > -1:
            examples = random.sample(examples,self._sample)

        for example in examples:
            question = example['question']['stem']
            choices = [c['text'] for c in example['question']['choices']]
            label = label_dict[example['answerKey']] if 'answerKey' in example else None
            yield self.text_to_instance(question, choices, label=label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         choices: List[str],
                         label: int = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        # we don't really need the [MASK] token with special brackets here, there is no notion of a [MASK] token position
        # in this baseline...
        question = question.replace('[MASK]','')

        question_tokens = self._tokenizer.tokenize(question)
        fields['phrase'] = TextField(question_tokens, self._token_indexers)

        choices_list = []
        for i, cho in enumerate(choices):
            choice_tokens = self._tokenizer.tokenize(cho)
            choices_list.append(TextField(choice_tokens, self._token_indexers))

        fields['choices'] = ListField(choices_list)

        metadata = {
            "question_text": question,
        #    "choice_text_list": choice_list,
        #    "correct_answer_index": answer_id,
        #    "question_tokens_list": qa_tokens_list,
        #    "choice_context_list": choice_context_list,
        #    "all_masked_index_ids": all_masked_index_ids
        }

        fields["metadata"] = MetadataField(metadata)

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
