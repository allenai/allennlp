import json
import logging
from collections import Counter
from typing import List, Dict

from overrides import overrides
from tqdm import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, IndexField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("squad")
class SquadReader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        instances = []
        for article in tqdm(dataset):
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    question_id = question_answer['id'].strip()

                    # There may be multiple answer annotations, so we pick the one that occurs the
                    # most.  This only matters on the SQuAD dev set, and it means our computed
                    # metrics ("start_acc", "end_acc", and "span_acc") aren't quite the same as the
                    # official metrics, which look at all of the annotations.  This is why we have
                    # a separate official SQuAD metric calculation (the "em" and "f1" metrics use
                    # the official script).
                    candidate_answers: Counter = Counter()
                    for answer in question_answer["answers"]:
                        candidate_answers[(answer["answer_start"], answer["text"])] += 1
                    answer_texts = [answer['text'] for answer in question_answer['answers']]
                    char_span_start, answer_text = candidate_answers.most_common(1)[0][0]

                    instance = self.text_to_instance(question_text,
                                                     paragraph,
                                                     question_id,
                                                     answer_text,
                                                     char_span_start,
                                                     tokenized_paragraph,
                                                     answer_texts)
                    instances.append(instance)
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         question_id: str = None,
                         answer_text: str = None,
                         char_span_start: int = None,
                         passage_tokens: List[Token] = None,
                         answer_texts: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        question_tokens = self._tokenizer.tokenize(question_text)
        # Separate so we can reference it later with a known type.
        passage_field = TextField(passage_tokens, self._token_indexers)
        fields['passage'] = passage_field
        fields['question'] = TextField(question_tokens, self._token_indexers)

        if answer_text:
            # SQuAD gives answer annotations as a character index into the paragraph, but we need a
            # token index for our models.  We convert them here.
            char_span_end = char_span_start + len(answer_text)
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start,
                                                                          char_span_end))
            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", passage_tokens)
                logger.debug("Question: %s", question_text)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", answer_text)

            fields['span_start'] = IndexField(span_start, passage_field)
            fields['span_end'] = IndexField(span_end, passage_field)
        metadata = {
                'original_passage': passage_text,
                'token_offsets': passage_offsets
                }
        if question_id:
            metadata['question_id'] = question_id
        if answer_texts:
            metadata['answer_texts'] = answer_texts
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'SquadReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers)
