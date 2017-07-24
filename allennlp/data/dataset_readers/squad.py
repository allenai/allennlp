from collections import Counter
import json
import logging
import random
from typing import List, Tuple, Dict, Set  # pylint: disable=unused-import

import numpy
from tqdm import tqdm

from allennlp.common import Params
from allennlp.data import Dataset, DatasetReader, Instance, TokenIndexer, Tokenizer
from allennlp.experiments import Registry
from allennlp.data.fields import TextField, ListField, IndexField
from allennlp.data.field import Field  # pylint: disable=unused-import
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Registry.register_dataset_reader("squad_sentence_selection")
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
    def from_params(cls, params: Params):
        """
        Parameters
        ----------
        squad_filename : ``str``
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
