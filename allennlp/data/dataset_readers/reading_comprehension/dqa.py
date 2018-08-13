import json
import logging
from typing import Any, Dict, List, Tuple


from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("dqa")
class DQAReader(DatasetReader):
    """
    Reads a JSON-formatted QUAC file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_text_lists'] and ``metadata['token_offsets']``.  

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    prev_a : Integer, optional 
        How many previous questions to consider.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 prev_a: int = 0,
                 add_query: bool = False,
                 prev_q_followup: bool =False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._prev_a = prev_a
        self._add_query = add_query
        self._prev_q_followup = prev_q_followup
    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        for article in dataset:
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)
                qas = paragraph_json['qas']
                metadata = {}
                metadata["instance_id"] = [qa['id'] for qa in qas]
                question_text_list = [qa["question"].strip().replace("\n", "") for qa in qas]
                answer_texts_list = [[answer['text'] for answer in qa['answers']] for qa in qas]
                metadata["question"] = question_text_list
                metadata['answer_texts_list'] = answer_texts_list
                span_starts_list = [[answer['answer_start'] for answer in qa['answers']] for qa in qas]
                span_ends_list = []
                for st_list, an_list in zip(span_starts_list, answer_texts_list):
                  span_ends = [start + len(answer) for start, answer in zip(st_list, an_list)]
                  span_ends_list.append(span_ends)
                yesno_list = [str(qa['yesno']) for qa in qas]
                followup_list = [str(qa['followup']) for qa in qas]
                instance = self.text_to_instance(question_text_list,
                                                 paragraph,
                                                 span_starts_list,
                                                 span_ends_list,
                                                 tokenized_paragraph,
                                                 yesno_list,
                                                 followup_list,
                                                 metadata)
                yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text_list: List[str],
                         passage_text: str,
                         start_span_list: List[List[int]] = None,
                         end_span_list: List[List[int]] = None,
                         passage_tokens: List[Token] = None,
                         yesno_list: List[int] = None,
                         followup_list: List[int] = None,
                         additional_metadata: Dict[str, Any] = None) -> Instance:
        # pylint: disable=arguments-differ
        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        answer_token_span_list = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for st_list, end_list in zip(start_span_list, end_span_list):
          token_spans: List[Tuple[int, int]] = []
          for char_span_start, char_span_end in zip(st_list, end_list):
              (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                           (char_span_start, char_span_end))
              if error:
                  logger.debug("Passage: %s", passage_text)
                  logger.debug("Passage tokens: %s", passage_tokens)
                  logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                  logger.debug("Token span: (%d, %d)", span_start, span_end)
                  logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                  logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
              token_spans.append((span_start, span_end))
          answer_token_span_list.append(token_spans)
        if self._add_query:
          added_question_text_list = []
          for q_idx, q in enumerate(question_text_list):
            if q_idx == 0 or self._prev_a == 0:
              added_question_text_list.append(q)
            else:
              if q_idx == 1 or self._prev_a == 1:
                added_question_text_list.append(question_text_list[q_idx-1]+"|||"+q)
              elif q_idx == 2 or self._prev_a == 2:
                added_question_text_list.append(question_text_list[q_idx-2]+"|-|"+question_text_list[q_idx-1]+"|||"+q)
              elif self._prev_a == 3:
                added_question_text_list.append(question_text_list[q_idx-3] +"||" + question_text_list[q_idx-2]+"|-|"+question_text_list[q_idx-1]+"|||"+q)
          question_text_list = added_question_text_list
        question_list_tokens = [self._tokenizer.tokenize(q) for q in question_text_list]
        additional_metadata['answer_texts_list'] = [util.handle_cannot(ans_list) for ans_list in additional_metadata['answer_texts_list']]
        return util.make_reading_comprehension_instance_dqa(question_list_tokens,
                                                            passage_tokens,
                                                            self._token_indexers,
                                                            passage_text,
                                                            answer_token_span_list,
                                                            yesno_list,
                                                            followup_list,
                                                            additional_metadata,
                                                            self._prev_a,
                                                            self._prev_q_followup)
