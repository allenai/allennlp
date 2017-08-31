import json
import logging
import random
from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Dict, Set  # pylint: disable=unused-import

import numpy
from tqdm import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.fields import TextField, ListField, IndexField
from allennlp.data.fields.field import Field  # pylint: disable=unused-import
from allennlp.data.tokenizers import WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _char_span_to_token_span(token_offsets: List[Tuple[int, int]],
                             character_span: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:
    """
    Converts a character span from a passage into the corresponding token span in the tokenized
    version of the passage.  If you pass in a character span that does not correspond to complete
    tokens in the tokenized version, we'll do our best, but the behavior is officially undefined.
    We return an error flag in this case, and have some debug logging so you can figure out the
    cause of this issue (in SQuAD, these are mostly either tokenization problems or annotation
    problems; there's a fair amount of both).

    The basic outline of this method is to find the token span that has the same offsets as the
    input character span.  If the tokenizer tokenized the passage correctly and has matching
    offsets, this is easy.  We try to be a little smart about cases where they don't match exactly,
    but mostly just find the closest thing we can.

    The returned ``(begin, end)`` indices are `inclusive` for both ``begin`` and ``end``.
    So, for example, ``(2, 2)`` is the one word span beginning at token index 2, ``(3, 4)`` is the
    two-word span beginning at token index 3, and so on.

    Returns
    -------
    token_span : ``Tuple[int, int]``
        `Inclusive` span start and end token indices that match as closely as possible to the input
        character spans.
    error : ``bool``
        Whether the token spans match the input character spans exactly.  If this is ``False``, it
        means there was an error in either the tokenization or the annotated character span.
    """
    # We have token offsets into the passage from the tokenizer; we _should_ be able to just find
    # the tokens that have the same offsets as our span.
    error = False
    start_index = 0
    while start_index < len(token_offsets) and token_offsets[start_index][0] < character_span[0]:
        start_index += 1
    # start_index should now be pointing at the span start index.
    if token_offsets[start_index][0] > character_span[0]:
        # In this case, a tokenization or labeling issue made us go too far - the character span
        # we're looking for actually starts in the previous token.  We'll back up one.
        logger.debug("Bad labelling or tokenization - start offset doesn't match")
        start_index -= 1
    if token_offsets[start_index][0] != character_span[0]:
        error = True
    end_index = start_index
    while end_index < len(token_offsets) and token_offsets[end_index][1] < character_span[1]:
        end_index += 1
    if end_index == start_index and token_offsets[end_index][1] > character_span[1]:
        # Looks like there was a token that should have been split, like "1854-1855", where the
        # answer is "1854".  We can't do much in this case, except keep the answer as the whole
        # token.
        logger.debug("Bad tokenization - end offset doesn't match")
    elif token_offsets[end_index][1] > character_span[1]:
        # This is a case where the given answer span is more than one token, and the last token is
        # cut off for some reason, like "split with Luckett and Rober", when the original passage
        # said "split with Luckett and Roberson".  In this case, we'll just keep the end index
        # where it is, and assume the intent was to mark the whole token.
        logger.debug("Bad labelling or tokenization - end offset doesn't match")
    if token_offsets[end_index][1] != character_span[1]:
        error = True
    return (start_index, end_index), error


@DatasetReader.register("squad")
class SquadReader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.

    The ``Instances`` also store their ID, the original passage text, and token offsets into the
    original passage in the instance metadata, accessible as ``instance.metadata['id']``,
    ``instance.metadata['original_passage']``, and ``instance.metadata['token_offsets']``.  This is
    so that we can more easily use the official SQuAD evaluation script to get metrics.

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
                 tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(tokenizer=tokenizer, token_indexers=token_indexers)

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

                # We add a special token to the end of the passage.  This is because our span
                # labels are end-exclusive, and we do a softmax over the passage to determine span
                # end.  So if we want to be able to include the last token of the passage, we need
                # to have a special symbol at the end.
                paragraph_tokens, paragraph_offsets = self._tokenizer.tokenize(paragraph)

                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    question_id = question_answer['id'].strip()
                    question_tokens, _ = self._tokenizer.tokenize(question_text)

                    # There may be multiple answer annotations, so pick the one that occurs the
                    # most.
                    candidate_answers = Counter() # type: Counter
                    for answer in question_answer["answers"]:
                        candidate_answers[(answer["answer_start"], answer["text"])] += 1
                    char_span_start, answer_text = candidate_answers.most_common(1)[0][0]

                    # SQuAD gives answer annotations as a character index into the paragraph, but
                    # we need a token index for our models.  We convert them here.
                    char_span_end = char_span_start + len(answer_text)
                    (span_start, span_end), error = _char_span_to_token_span(paragraph_offsets,
                                                                             (char_span_start,
                                                                              char_span_end))
                    if error:
                        logger.debug("Passage: %s", paragraph)
                        logger.debug("Passage tokens: %s", paragraph_tokens)
                        logger.debug("Question: %s", question_text)
                        logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                        logger.debug("Token span: (%d, %d)", span_start, span_end)
                        logger.debug("Tokens in answer: %s", paragraph_tokens[span_start:span_end + 1])
                        logger.debug("Answer: %s", answer_text)

                    # Because the paragraph is shared across multiple questions, we do a deepcopy
                    # here to avoid any weird issues with shared state between instances (e.g.,
                    # when indexing is done, and when padding is done).  I _think_ all of those
                    # operations would be safe with shared objects, but I'd rather just be safe by
                    # doing a copy here.  Extra memory usage should be minimal.
                    paragraph_field = TextField(deepcopy(paragraph_tokens), self._token_indexers)
                    question_field = TextField(question_tokens, self._token_indexers)
                    span_start_field = IndexField(span_start, paragraph_field)
                    span_end_field = IndexField(span_end, paragraph_field)
                    fields = {
                            'question': question_field,
                            'passage': paragraph_field,
                            'span_start': span_start_field,
                            'span_end': span_end_field
                            }
                    metadata = {
                            'id': question_id,
                            'original_passage': paragraph,
                            'token_offsets': paragraph_offsets
                            }
                    instance = Instance(fields, metadata)
                    instances.append(instance)
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @classmethod
    def from_params(cls, params: Params) -> 'SquadReader':
        """
        Parameters
        ----------
        tokenizer : ``Params``, optional (default=``{}``)
        token_indexers: ``Params``, optional (default=``{}``)
        """
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = {}
        token_indexer_params = params.pop('token_indexers', {})
        for name, indexer_params in token_indexer_params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)
        # The default parameters are contained within the class, so if no parameters are given we
        # must pass None.
        if token_indexers == {}:
            token_indexers = None
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers)


@DatasetReader.register("squad_sentence_selection")
class SquadSentenceSelectionReader(DatasetReader):
    """
    Parameters
    ----------
    negative_sentence_selection : ``str``, optional (default="paragraph")
        A comma-separated list of methods to use to generate negative sentences in the data.

        There are three options here:

        (1) "paragraph", which means to use as negative sentences all other sentences in the same
            paragraph as the correct answer sentence.
        (2) "random-[int]", which means to randomly select [int] sentences from all SQuAD sentences
            to use as negative sentences.
        (3) "pad-to-[int]", which means to randomly select sentences from all SQuAD sentences until
            there are a total of [int] sentences.  This will not remove any previously selected
            sentences if you already have more than [int].
        (4) "question", which means to use as a negative sentence the `question` itself.
        (5) "questions-random-[int]", which means to select [int] random `questions` from SQuAD to
            use as negative sentences (this could include the question corresponding to the
            example; we don't filter out that case).

        We will process these options in order, so the "pad-to-[int]" option mostly only makes
        sense as the last option.
    tokenizer : ``Tokenizer``, optional
        We use this ``Tokenizer`` for both the question and the sentences.  See :class:`Tokenizer`.
        Default is ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the sentences.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 negative_sentence_selection: str = "paragraph",
                 tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(tokenizer=tokenizer, token_indexers=token_indexers)
        self._negative_sentence_selection_methods = negative_sentence_selection.split(",")

        # Initializing some data structures here that will be useful when reading a file.
        # Maps sentence strings to sentence indices
        self._sentence_to_id = {}  # type: Dict[str, int]
        # Maps sentence indices to sentence strings
        self._id_to_sentence = {}  # type: Dict[int, str]
        # Maps paragraph ids to lists of contained sentence ids
        self._paragraph_sentences = {}  # type: Dict[int, List[int]]
        # Maps sentence ids to the containing paragraph id.
        self._sentence_paragraph_map = {}  # type: Dict[int, int]
        # Maps question strings to question indices
        self._question_to_id = {}  # type: Dict[str, int]
        # Maps question indices to question strings
        self._id_to_question = {}  # type: Dict[int, str]

    def _get_sentence_choices(self,
                              question_id: int,
                              answer_id: int) -> Tuple[List[str], int]:  # pylint: disable=invalid-sequence-index
        # Because sentences and questions have different indices, we need this to hold tuples of
        # ("sentence", id) or ("question", id), instead of just single ids.
        negative_sentences = set()  # type: Set[Tuple[str, int]]
        for selection_method in self._negative_sentence_selection_methods:
            if selection_method == 'paragraph':
                paragraph_id = self._sentence_paragraph_map[answer_id]
                paragraph_sentences = self._paragraph_sentences[paragraph_id]
                negative_sentences.update(("sentence", sentence_id)
                                          for sentence_id in paragraph_sentences
                                          if sentence_id != answer_id)
            elif selection_method.startswith("random-"):
                num_to_pick = int(selection_method.split('-')[1])
                num_sentences = len(self._sentence_to_id)
                # We'll ignore here the small probability that we pick `answer_id`, or a
                # sentence we've chosen previously.
                selected_ids = numpy.random.choice(num_sentences, (num_to_pick,), replace=False)
                negative_sentences.update(("sentence", sentence_id)
                                          for sentence_id in selected_ids
                                          if sentence_id != answer_id)
            elif selection_method.startswith("pad-to-"):
                desired_num_sentences = int(selection_method.split('-')[2])
                # Because we want to pad to a specific number of sentences, we'll do the choice
                # logic in a loop, to be sure we actually get to the right number.
                while desired_num_sentences > len(negative_sentences):
                    num_to_pick = desired_num_sentences - len(negative_sentences)
                    num_sentences = len(self._sentence_to_id)
                    if num_to_pick > num_sentences:
                        raise RuntimeError("Not enough sentences to pick from")
                    selected_ids = numpy.random.choice(num_sentences, (num_to_pick,), replace=False)
                    negative_sentences.update(("sentence", sentence_id)
                                              for sentence_id in selected_ids
                                              if sentence_id != answer_id)
            elif selection_method == "question":
                negative_sentences.add(("question", question_id))
            elif selection_method.startswith("questions-random-"):
                num_to_pick = int(selection_method.split('-')[2])
                num_questions = len(self._question_to_id)
                # We'll ignore here the small probability that we pick `question_id`, or a
                # question we've chosen previously.
                selected_ids = numpy.random.choice(num_questions, (num_to_pick,), replace=False)
                negative_sentences.update(("question", q_id) for q_id in selected_ids)
            else:
                raise RuntimeError("Unrecognized selection method:", selection_method)
        choices = list(negative_sentences) + [("sentence", answer_id)]
        random.shuffle(choices)
        correct_choice = choices.index(("sentence", answer_id))
        sentence_choices = []
        for sentence_type, index in choices:
            if sentence_type == "sentence":
                sentence_choices.append(self._id_to_sentence[index])
            else:
                sentence_choices.append(self._id_to_question[index])
        return sentence_choices, correct_choice

    def read(self, file_path: str):
        # Import is here, since it isn't necessary by default.
        import nltk

        # Holds tuples of (question_text, answer_sentence_id)
        questions = []
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        for article in tqdm(dataset):
            for paragraph in article['paragraphs']:
                paragraph_id = len(self._paragraph_sentences)
                self._paragraph_sentences[paragraph_id] = []

                context_article = paragraph["context"]

                # Split the context_article into a list of sentences.
                sentences = nltk.sent_tokenize(context_article)

                # Make a dict from span indices to sentence. The end span is
                # exclusive, and the start span is inclusive.
                span_to_sentence_index = {}
                current_index = 0
                for sentence in sentences:
                    sentence_id = len(self._sentence_to_id)
                    self._sentence_to_id[sentence] = sentence_id
                    self._id_to_sentence[sentence_id] = sentence
                    self._sentence_paragraph_map[sentence_id] = paragraph_id
                    self._paragraph_sentences[paragraph_id].append(sentence_id)

                    sentence_len = len(sentence)
                    # Need to add one to the end index to account for the
                    # trailing space after punctuation that is stripped by NLTK.
                    span_to_sentence_index[(current_index,
                                            current_index + sentence_len + 1)] = sentence
                    current_index += sentence_len + 1
                for question_answer in paragraph['qas']:
                    question_text = question_answer["question"].strip()
                    question_id = len(self._question_to_id)
                    self._question_to_id[question_text] = question_id
                    self._id_to_question[question_id] = question_text

                    # There may be multiple answer annotations, so pick the one
                    # that occurs the most.
                    candidate_answer_start_indices = Counter() # type: Counter
                    for answer in question_answer["answers"]:
                        candidate_answer_start_indices[answer["answer_start"]] += 1
                    answer_start_index, _ = candidate_answer_start_indices.most_common(1)[0]

                    # Get the full sentence corresponding to the answer.
                    answer_sentence = None
                    for span_tuple in span_to_sentence_index:
                        start_span, end_span = span_tuple
                        if start_span <= answer_start_index and answer_start_index < end_span:
                            answer_sentence = span_to_sentence_index[span_tuple]
                            break
                    else:  # no break
                        raise ValueError("Index of answer start was out of bounds. "
                                         "This should never happen, please raise "
                                         "an issue on GitHub.")

                    # Now that we have the string of the full sentence, we need to
                    # search for it in our shuffled list to get the index.
                    answer_id = self._sentence_to_id[answer_sentence]

                    # Now we can make the string representation and add this
                    # to the list of processed_rows.
                    questions.append((question_id, answer_id))
        instances = []
        logger.info("Processing questions into training instances")
        for question_id, answer_id in tqdm(questions):
            sentence_choices, correct_choice = self._get_sentence_choices(question_id, answer_id)
            question_text = self._id_to_question[question_id]
            sentence_fields = []  # type: List[Field]
            for sentence in sentence_choices:
                tokenized_sentence, _ = self._tokenizer.tokenize(sentence)
                sentence_field = TextField(tokenized_sentence, self._token_indexers)
                sentence_fields.append(sentence_field)
            sentences_field = ListField(sentence_fields)
            question_tokens, _ = self._tokenizer.tokenize(question_text)
            question_field = TextField(question_tokens, self._token_indexers)
            correct_sentence_field = IndexField(correct_choice, sentences_field)
            instances.append(Instance({'question': question_field,
                                       'sentences': sentences_field,
                                       'correct_sentence': correct_sentence_field}))

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @classmethod
    def from_params(cls, params: Params) -> 'SquadSentenceSelectionReader':
        """
        Parameters
        ----------
        negative_sentence_selection : ``str``, optional (default="paragraph")
        tokenizer : ``Params``, optional
        token_indexers: ``List[Params]``, optional
        """
        negative_sentence_selection = params.pop('negative_sentence_selection', 'paragraph')
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
        return SquadSentenceSelectionReader(negative_sentence_selection=negative_sentence_selection,
                                            tokenizer=tokenizer,
                                            token_indexers=token_indexers)
