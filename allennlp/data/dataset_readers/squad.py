import json
import logging
import random
from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Dict, Set  # pylint: disable=unused-import

import numpy
from tqdm import tqdm

from allennlp.common import Params
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.fields import TextField, ListField, IndexField
from allennlp.data.fields.field import Field  # pylint: disable=unused-import
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _char_span_to_token_span(sentence: str,
                             tokenized_sentence: List[str],
                             span: Tuple[int, int],
                             tokenizer: Tokenizer,
                             slack: int = 3) -> Tuple[int, int]:
    """
    Converts a character span from a sentence into the corresponding token span in the
    tokenized version of the sentence.  If you pass in a character span that does not
    correspond to complete tokens in the tokenized version, we'll do our best, but the behavior
    is officially undefined.

    The basic outline of this method is to find the token that starts the same number of
    characters into the sentence as the given character span.  We try to handle a bit of error
    in the tokenization by checking `slack` tokens in either direction from that initial
    estimate.

    The returned ``(begin, end)`` indices are `inclusive` for ``begin``, and `exclusive` for
    ``end``.  So, for example, ``(2, 2)`` is an empty span, ``(2, 3)`` is the one-word span
    beginning at token index 2, and so on.
    """
    # First we'll tokenize the span and the sentence, so we can count tokens and check for
    # matches.
    span_chars = sentence[span[0]:span[1]]
    tokenized_span = tokenizer.tokenize(span_chars)
    # Then we'll find what we think is the first token in the span
    chars_seen = 0
    index = 0
    while index < len(tokenized_sentence) and chars_seen < span[0]:
        chars_seen += len(tokenized_sentence[index]) + 1
        index += 1
    # index is now the span start index.  Is it a match?
    if _spans_match(tokenized_sentence, tokenized_span, index):
        return (index, index + len(tokenized_span))
    for i in range(1, slack + 1):
        if _spans_match(tokenized_sentence, tokenized_span, index + i):
            return (index + i, index + i+ len(tokenized_span))
        if _spans_match(tokenized_sentence, tokenized_span, index - i):
            return (index - i, index - i + len(tokenized_span))
    # No match; we'll just return our best guess.
    return (index, index + len(tokenized_span))


def _spans_match(sentence_tokens: List[str], span_tokens: List[str], index: int) -> bool:
    if index < 0 or index >= len(sentence_tokens):
        return False
    if sentence_tokens[index] == span_tokens[0]:
        span_index = 1
        while (span_index < len(span_tokens) and
               sentence_tokens[index + span_index] == span_tokens[span_index]):
            span_index += 1
        if span_index == len(span_tokens):
            return True
    return False


@DatasetReader.register("squad")
class SquadReader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
    """
    def __init__(self,
                 tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def read(self, file_path: str):

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        instances = []
        for article in tqdm(dataset):
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]
                # replace newlines in the paragraph
                cleaned_paragraph = paragraph.replace("\n", " ")

                # We add a special token to the end of the passage.  This is because our span
                # labels are end-exclusive, and we do a softmax over the passage to determine span
                # end.  So if we want to be able to include the last token of the passage, we need
                # to have a special symbol at the end.
                tokenized_paragraph = self._tokenizer.tokenize(cleaned_paragraph) + ['@@STOP@@']

                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    tokenized_question = self._tokenizer.tokenize(question_text)

                    # There may be multiple answer annotations, so pick the one that occurs the
                    # most.
                    candidate_answers = Counter() # type: Counter
                    for answer in question_answer["answers"]:
                        candidate_answers[(answer["answer_start"], answer["text"])] += 1
                    char_span_start, answer_text = candidate_answers.most_common(1)[0][0]

                    # SQuAD gives answer annotations as a character index into the paragraph, but
                    # we need a token index for our models.  We convert them here.
                    char_span_end = char_span_start + len(answer_text)
                    span_start, span_end = _char_span_to_token_span(paragraph,
                                                                    tokenized_paragraph,
                                                                    (char_span_start, char_span_end),
                                                                    self._tokenizer)

                    # Because the paragraph is shared across multiple questions, we do a deepcopy
                    # here to avoid any weird issues with shared state between instances (e.g.,
                    # when indexing is done, and when padding is done).  I _think_ all of those
                    # operations would be safe with shared objects, but I'd rather just be safe by
                    # doing a copy here.  Extra memory usage should be minimal.
                    paragraph_field = TextField(deepcopy(tokenized_paragraph), self._token_indexers)
                    question_field = TextField(tokenized_question, self._token_indexers)
                    span_start_field = IndexField(span_start, paragraph_field)
                    span_end_field = IndexField(span_end, paragraph_field)
                    instance = Instance({
                            'question': question_field,
                            'passage': paragraph_field,
                            'span_start': span_start_field,
                            'span_end': span_end_field
                            })
                    instances.append(instance)
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
    negative_sentence_selection : ``str``, optional (default=``"paragraph"``)
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
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the sentences.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the question and the sentences.  See :class:`TokenIndexer`.
    """
    def __init__(self,
                 negative_sentence_selection: str = "paragraph",
                 tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._negative_sentence_selection_methods = negative_sentence_selection.split(",")
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

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
                # replace newlines in the context article
                cleaned_context_article = context_article.replace("\n", "")

                # Split the cleaned_context_article into a list of sentences.
                sentences = nltk.sent_tokenize(cleaned_context_article)

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
                tokenized_sentence = self._tokenizer.tokenize(sentence)
                sentence_field = TextField(tokenized_sentence, self._token_indexers)
                sentence_fields.append(sentence_field)
            sentences_field = ListField(sentence_fields)
            tokenized_question = self._tokenizer.tokenize(question_text)
            question_field = TextField(tokenized_question, self._token_indexers)
            correct_sentence_field = IndexField(correct_choice, sentences_field)
            instances.append(Instance({'question': question_field,
                                       'sentences': sentences_field,
                                       'correct_sentence': correct_sentence_field}))
        return Dataset(instances)

    @classmethod
    def from_params(cls, params: Params) -> 'SquadSentenceSelectionReader':
        """
        Parameters
        ----------
        negative_sentence_selection : ``str``, optional (default=``"paragraph"``)
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
