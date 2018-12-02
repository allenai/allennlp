import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile,os

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("multiqa")
class MultiQAReader(DatasetReader):
    """
    Reads a JSON-formatted Quesiton Answering in Context (QuAC) data file
    and returns a ``Dataset`` where the ``Instances`` have four fields: ``question``, a ``ListField``,
    ``passage``, another ``TextField``, and ``span_start`` and ``span_end``, both ``ListField`` composed of
    IndexFields`` into the ``passage`` ``TextField``.
    Two ``ListField``, composed of ``LabelField``, ``yesno_list`` and  ``followup_list`` is added.
    We also add a
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
    num_context_answers : ``int``, optional
        How many previous question answers to consider in a context.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 num_context_answers: int = 0,
                 max_context_size: int = 5000,
                 num_of_examples_to_sample: int = None,
                 use_document_titles:bool= False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._num_context_answers = num_context_answers
        self._max_context_size = max_context_size
        self._use_document_titles = use_document_titles
        self._num_of_examples_to_sample = num_of_examples_to_sample

    @overrides
    def _read(self, file_path: str):
        logger.info("Reading the dataset")

        # supporting multi dataset training:
        contexts = []
        for single_file_path in file_path.split(','):
            # if `file_path` is a URL, redirect to the cache
            single_file_path = cached_path(single_file_path)
            logger.info("Reading file at %s", single_file_path)

            with zipfile.ZipFile(single_file_path, 'r') as myzip:
                with myzip.open(myzip.namelist()[0]) as myfile:
                    dataset_json = json.load(myfile)
                    contexts += dataset_json['data']['contexts']

        skipped_qa_count = 0
        all_qa_count = 0

        if self._num_of_examples_to_sample is not None:
            contexts = contexts[0:self._num_of_examples_to_sample]

        for context_ind,context in enumerate(contexts):

            # Processing each document separatly
            paragraph = ''
            tokenized_paragraph = []
            answer_starts_offsets = []
            offset = 0
            for doc_ind, document in enumerate(context['documents']):
                # tokenizing the whole document (title + all snippets concatinated)
                ## TODO add document['rank']
                # constracting single context by concatinating parts of the original context
                if self._use_document_titles:
                    text_to_add =  document['title'] + ' | ' + ' '.join(document['snippets']) + " || "
                else:
                    text_to_add = ' '.join(document['snippets']) + " || "

                # Stop when context is larger than max size.
                tokens_to_add  = self._tokenizer.tokenize(text_to_add)
                if len(tokenized_paragraph) + len(tokens_to_add) > self._max_context_size:
                    break

                tokenized_paragraph += tokens_to_add
                paragraph += text_to_add

                # Computing answer_starts offsets:
                if self._use_document_titles:
                    answer_starts_offsets.append({'title': offset})
                    offset += len(document['title']) + 3  # we add 3 for the separator ' | '
                else:
                    answer_starts_offsets.append({})

                for snippet_ind, snippet in enumerate(document['snippets']):
                    answer_starts_offsets[doc_ind][snippet_ind] = offset
                    offset += len(snippet)

                    # ' '. adds extra space between the snippets.
                    if len(document['snippets'])>1 and snippet_ind < len(document['snippets'])-1:
                        offset += 1
                offset += 4 # for " || "

                if offset != len(paragraph):
                    raise ValueError()

            # we need to tokenize all the paragraph (again) because previous tokens start the offset count
            # from 0 for each document... # TODO find a better way to do this...
            tokenized_paragraph = self._tokenizer.tokenize(paragraph)

            # Discarding context that are too long (when len is 0 that means we breaked from context loop)
            all_qa_count += len(context['qas'])
            if len(tokenized_paragraph) > self._max_context_size or len(tokenized_paragraph) == 0:
                skipped_qa_count += len(context['qas'])
                if context_ind % 30 == 0:
                    logger.info('Fraction of QA remaining = %f', ((all_qa_count - skipped_qa_count) / all_qa_count))
                continue



            # a list of question/answers
            qas = context['qas']


            # Adding Metadata
            metadata = {}
            metadata["instance_id"] = [qa['id'] for qa in qas]
            question_text_list = [qa["question"].strip().replace("\n", "") for qa in qas]
            answer_texts_list = [[] for qa in qas]
            for qa_ind,qa in enumerate(qas):
                for answer in qa['answers']:
                    answer_texts_list[qa_ind] += [alias['text'] for alias in answer['aliases']]
            metadata["question"] = question_text_list
            metadata['answer_texts_list'] = answer_texts_list

            # calculate new answer starts for the new combined document
            span_starts_list = {'answers':[[] for qa in qas],'distractor_answers':[[] for qa in qas]}
            span_ends_list = {'answers':[[] for qa in qas],'distractor_answers':[[] for qa in qas]}
            for qa_ind, qa in enumerate(qas):
                if qa['answer_type'] == 'multi_choice':
                    answer_types = ['answers','distractor_answers']
                else:
                    answer_types = ['answers']

                for answer_type in answer_types:
                    #span_starts_list[answer_type] = [[] for qa in qas]
                    #span_ends_list[answer_type] = [[] for qa in qas]
                    for answer in qa[answer_type]:
                        for alias in answer['aliases']:
                            for alias_start in alias['answer_starts']:
                                # It's possible we didn't take all the contexts.
                                if len(answer_starts_offsets) > alias_start[0] and \
                                        alias_start[1] in answer_starts_offsets[alias_start[0]] and  \
                                        (alias_start[1] != 'title' or self._use_document_titles):
                                    answer_start_norm = answer_starts_offsets[alias_start[0]][alias_start[1]] + alias_start[2]
                                    span_starts_list[answer_type][qa_ind].append(answer_start_norm)
                                    span_ends_list[answer_type][qa_ind].append(answer_start_norm + len(alias['text']))

                                    # Sanity check: the alias text should be equal the text in answer_start in the paragraph
                                    import re
                                    x = re.match(r'\b{0}\b'.format(re.escape(alias['text'])),
                                                 paragraph[answer_start_norm:answer_start_norm + len(alias['text'])],
                                                 re.IGNORECASE)
                                    if x is None:
                                        if (alias['text'] != paragraph[answer_start_norm:answer_start_norm + len(alias['text'])]):
                                            raise ValueError("answers and paragraph not aligned!")

            # If answer was not found in this question do not yield an instance
            # (This could happen if we used part of the context or in unfiltered context versions)
            if span_starts_list['answers'] == [[]]:
                skipped_qa_count += len(context['qas'])
                if context_ind % 20 == 0:
                    logger.info('Fraction of QA remaining = %f', ((all_qa_count - skipped_qa_count) / all_qa_count))
                continue

            instance = self.text_to_instance(question_text_list,
                                             paragraph,
                                             span_starts_list['answers'],
                                             span_ends_list['answers'],
                                             tokenized_paragraph,
                                             metadata)

            # passing the tokens of all gold answer instances (not just the "original answer"
            # as in the original quac dataset reader)
            passage_offsets = [(token.idx, token.idx + len(token.text)) for token in tokenized_paragraph]
            instance.fields['metadata'].metadata['gold_answer_start_list'] = []
            instance.fields['metadata'].metadata['gold_answer_end_list'] = []
            for span_char_start,span_char_end in zip(span_starts_list['answers'][0],span_ends_list['answers'][0]):
                (span_start, span_end), error = util.char_span_to_token_span(passage_offsets, \
                                                                             (span_char_start, span_char_end))

                instance.fields['metadata'].metadata['gold_answer_start_list'].append(span_start)
                instance.fields['metadata'].metadata['gold_answer_end_list'].append(span_end)

            # Multiple choice answer support. (all instance of other answers should be
            # contained in "multichoice_incorrect_answers", same format as "answers" field)
            if qas[0]['answer_type'] == 'multi_choice':
                instance.fields['metadata'].metadata['multichoice_incorrect_answers_start_list'] = []
                instance.fields['metadata'].metadata['multichoice_incorrect_answers_end_list'] = []
                for span_char_start, span_char_end in \
                        zip(span_starts_list['distractor_answers'][0], span_ends_list['distractor_answers'][0]):
                    (span_start, span_end), error = util.char_span_to_token_span(passage_offsets, \
                                                                                 (span_char_start, span_char_end))

                    instance.fields['metadata'].metadata['multichoice_incorrect_answers_start_list'].append(span_start)
                    instance.fields['metadata'].metadata['multichoice_incorrect_answers_end_list'].append(span_end)

            yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text_list: List[str],
                         passage_text: str,
                         start_span_list: List[List[int]] = None,
                         end_span_list: List[List[int]] = None,
                         passage_tokens: List[Token] = None,
                         additional_metadata: Dict[str, Any] = None) -> Instance:
        # pylint: disable=arguments-differ
        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        answer_token_span_list = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for start_list, end_list in zip(start_span_list, end_span_list):
            token_spans: List[Tuple[int, int]] = []
            for char_span_start, char_span_end in zip(start_list, end_list):
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
        question_list_tokens = [self._tokenizer.tokenize(q) for q in question_text_list]
        # Map answer texts to "CANNOTANSWER" if more than half of them marked as so.
        additional_metadata['answer_texts_list'] = [util.handle_cannot(ans_list) for ans_list \
                                                    in additional_metadata['answer_texts_list']]
        return util.make_reading_comprehension_instance_multiqa(question_list_tokens,
                                                             passage_tokens,
                                                             self._token_indexers,
                                                             passage_text,
                                                             answer_token_span_list,
                                                             additional_metadata,
                                                             self._num_context_answers)
