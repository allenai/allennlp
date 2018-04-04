# pylint: disable=no-member
import json
import logging
import pathlib
import shutil
import tarfile
import tempfile
from typing import Dict, List, Tuple, Iterator, Iterable, NamedTuple

from overrides import overrides
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np

from allennlp.common import Params, JsonDict
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.token import json_to_token, token_to_json, truncate_token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

_PARAGRAPH_TOKEN = "@@PARAGRAPH@@"

# We create NamedTuples for storing "merged paragraphs" and questions
# to make it easier to preprocess the dataset into JSONL format
# and start training from the processed data.

class MergedParagraphs(NamedTuple):
    """
    A typical TriviaQA question has multiple files with answers, each consisting
    of many small paragraphs. We merge these into fewer paragraphs of larger size.
    """
    texts: List[str]
    tokens: List[List[Token]]
    token_spans: List[List[Tuple[int, int]]]
    has_answers: List[int] = None

    def to_json(self) -> JsonDict:
        return {
                "texts": self.texts,
                "tokens": [[token_to_json(token) for token in tokens_i]
                           for tokens_i in self.tokens],
                "token_spans": self.token_spans,
                "has_answers": self.has_answers
        }

    @staticmethod
    def from_json(blob: JsonDict) -> 'MergedParagraphs':
        return MergedParagraphs(texts=blob['texts'],
                                tokens=[[json_to_token(b) for b in tokens_i]
                                        for tokens_i in blob['tokens']],
                                token_spans=blob['token_spans'],
                                has_answers=blob.get('has_answers'))


class Question(NamedTuple):
    """
    We process each TriviaQA question into tokenized question text
    and merged paragraphs that may or may not contain the answer.
    """
    id: str  # pylint: disable=invalid-name
    text: str
    tokens: List[Token]
    paragraphs: MergedParagraphs
    answer_texts: List[str] = None

    def to_json(self) -> JsonDict:
        return {
                "id": self.id,
                "text": self.text,
                "tokens": [token_to_json(token) for token in self.tokens],
                "paragraphs": self.paragraphs.to_json(),
                "answer_texts": self.answer_texts
        }

    @staticmethod
    def from_json(blob: JsonDict) -> 'Question':
        return Question(id=blob['id'],
                        text=blob['text'],
                        tokens=[json_to_token(token) for token in blob['tokens']],
                        paragraphs=MergedParagraphs.from_json(blob['paragraphs']),
                        answer_texts=blob.get('answer_texts'))


def _merge(paragraphs: List[str],
           question: str,
           answer_texts: List[str],
           tokenizer: Tokenizer,
           max_paragraphs: int = 4,
           tfidf_sort: bool = True,
           require_answer: bool = False,
           max_paragraph_length: int = 400) -> MergedParagraphs:
    """
    Given some (small?) paragraphs and a question, merge the paragraphs to make
    synthetic paragraphs of size ``max_paragraph_length``, and return a ``MergedParagraphs``
    instance containing the "best" ``topn`` of them. If ``topn`` is None, return all of them.

    If ``tfidf_sort`` is True, "best" means closest (in terms of tfidf distance) to the question.
    If ``require_answer`` is True, then the best "best" paragraph is also required to contain
    one of the ``answer_texts``. Don't do this with a test or validation dataset, that's cheating.
    If neither is True, then just the first ``topn`` paragraphs are returned.
    """
    # Collect all the text into one mega-paragraph.
    tokens: List[str] = []
    for paragraph in paragraphs:
        tokens.extend(token.text for token in tokenizer.tokenize(paragraph))
        tokens.append(_PARAGRAPH_TOKEN)

    # Get rid of trailing paragraph token
    tokens = tokens[:-1]

    # Merge them into paragraphs of size ``max_paragraph_length``.
    merged_paragraphs = [' '.join(paragraph_tokens)
                         for paragraph_tokens in lazy_groups_of(iter(tokens), max_paragraph_length)]
    merged_paragraph_tokens = [tokenizer.tokenize(paragraph) for paragraph in merged_paragraphs]

    # If ``max_paragraphs`` is None, we want all the merged paragraphs.
    if max_paragraphs is None:
        max_paragraphs = len(merged_paragraphs)

    # Get the ranked indexes, from "best" paragraph to worst paragraph.
    if tfidf_sort:
        # Sort the paragraphs by their tfidf scores with the question.
        try:
            tfidf = TfidfVectorizer(strip_accents="unicode", stop_words="")
            paragraph_features = tfidf.fit_transform(merged_paragraphs)
            question_features = tfidf.transform([question])
            scores = pairwise_distances(question_features, paragraph_features, "cosine").ravel()
        except ValueError:
            scores = np.array([0.0] * len(merged_paragraphs))

        # Sort by distance, break ties by original order.
        ranks: Iterable[int] = [i for i, _ in sorted(enumerate(scores), key=lambda pair: (pair[1], pair[0]))]
    else:
        # Keep the paragraphs in their original order.
        ranks = range(len(merged_paragraphs))

    # Find the indices of paragraphs that have answers.
    has_answers = [i for i in ranks
                   if util.find_valid_answer_spans(merged_paragraph_tokens[i], answer_texts)]

    # In the require_answer case, we just return None if no paragraph has the answer.
    if require_answer and not has_answers:
        return None

    if require_answer:
        # Want first_answer to be the first paragraph, and then take the most highly ranked
        # other topn - 1
        first_answer = has_answers[0]
        choices: Iterable[int] = ([first_answer] +
                                  [i for i in ranks if i != first_answer][:(max_paragraphs - 1)])
    else:
        choices = range(min(max_paragraphs, len(merged_paragraphs)))

    merged_texts = [merged_paragraphs[i] for i in choices]
    merged_tokens = [merged_paragraph_tokens[i] for i in choices]
    merged_token_spans = [util.find_valid_answer_spans(tokens_i, answer_texts)
                          for tokens_i in merged_tokens]

    return MergedParagraphs(texts=merged_texts,
                            tokens=merged_tokens,
                            token_spans=merged_token_spans,
                            has_answers=[i for i, choice in enumerate(choices) if choice in has_answers])


def process_triviaqa_questions(evidence_path: pathlib.Path,
                               questions_path: pathlib.Path,
                               tokenizer: Tokenizer,
                               max_paragraphs: int = 4,
                               tfidf_sort: bool = True,
                               require_answer: bool = False,
                               max_paragraph_length: int = 400) -> Iterator[Question]:
    """
    Processes one of the questions files in the (untarred) TriviaQA dataset
    into ``Question`` objects. For each question in the specified JSON file,
    we load all of the evidence files, merge the paragraphs to get new paragraphs
    of size ``max_paragraph_length``, sort them by tfidf distance to the question,
    and take the ``topn`` closest paragraphs.
    """
    if 'web' in questions_path.name:
        result_key, evidence_subdir = 'SearchResults', 'web'
    else:
        result_key, evidence_subdir = 'EntityPages', 'wikipedia'

    with open(questions_path, 'r') as f:
        questions_data = json.loads(f.read())['Data']

    for question in questions_data:
        question_id = question['QuestionId']
        question_text = question['Question']
        question_tokens = tokenizer.tokenize(question_text)

        answer = question['Answer']
        human_answers = [util.normalize_text(human_answer)
                         for human_answer in answer.get('HumanAnswers', [])]
        answer_texts = answer['NormalizedAliases'] + human_answers
        evidence_files = [result['Filename'] for result in question[result_key]]

        paragraphs: List[str] = []

        for evidence_file in evidence_files:
            evidence_file_path = evidence_path / 'evidence' / evidence_subdir / evidence_file
            with open(evidence_file_path, 'r') as evidence_file:
                paragraphs.extend(evidence_file.readlines())

        merged_paragraphs = _merge(paragraphs=paragraphs,
                                   question=question_text,
                                   answer_texts=answer_texts,
                                   tokenizer=tokenizer,
                                   max_paragraphs=max_paragraphs,
                                   tfidf_sort=tfidf_sort,
                                   require_answer=require_answer,
                                   max_paragraph_length=max_paragraph_length)

        if not merged_paragraphs:
            continue

        question = Question(id=question_id,
                            text=question_text,
                            tokens=question_tokens,
                            paragraphs=merged_paragraphs,
                            answer_texts=answer_texts)

        yield question

@DatasetReader.register("triviaqa")
class TriviaQaReader(DatasetReader):
    """
    Reads the TriviaQA dataset into a ``Dataset`` containing ``Instances`` with three fields:
    ``question`` (a ``TextField``), ``paragraphs`` (a ``ListField[TextField]``),
    and ``spans`` (a ``ListField[SpanField]``). Each instance consists of one or more paragraphs
    from the same document, chosen in a manner specified by the ``paragraph_picker`` parameter.

    TriviaQA is split up into several JSON files defining the questions, and a lot of text files
    containing crawled web documents. These can be read from the tarball, from the unzipped
    tarball, or from preprocessed JSON files. (The preprocessing is expensive, so if you want
    to run a lot of experiments you should try to only do it once.)

    Because we need to read both train and validation files from the same tarball, we take the
    tarball itself as a constructor parameter, and take the question file as the argument to
    ``read``.  This means that you should give the path to the tarball in the ``dataset_reader``
    parameters in your experiment configuration file, and something like ``"wikipedia-train.json"``
    for the ``train_data_path`` and ``validation_data_path``.

    Parameters
    ----------
    triviaqa_path : ``str``
        The path to the TriviaQA data, which may be in one of several formats.
    unfiltered_path : ``str``, optional
        The path to the "unfiltered" TriviaQA data, which contains only the questions
        and points to evidence files under the original triviaqa_path.
    data_format : ``str``, optional (default: "tar.gz")
        Valid values are ``"tar.gz"``, for the single file you can download from the TriviaQA website,
        ``"unpackaged"``, for if you've extracted the tar file into a directory, or
        ``"preprocessed"``, for if you've already preprocessed it into serialized ``Question`` s.
    paragraph_picker: ``str``, optional, (default: ``None``)
        If specified, this indicates the scheme for sampling paragraphs
        for each question-document pair. Currently, the only supported
        option is ``triviaqa-web-train``.
    sample_first_iteration: ``bool``, optional, (default: ``False``)
        Depending on the value of ``paragraph_picker``, the dataset reader
        may return a random sample of paragraphs each iteration. This flag
        disables the sampling during the first iteration, which is likely to
        be the iteration for creating the ``Vocabulary``, in which you would
        like all the paragraphs to be represented. However, if you are using
        a precomputed ``Vocabulary``, you should set this to ``True``.
    keep_questions_in_memory: ``bool``, optional, (default: ``True``)
        The training will run faster (and use more memory) if you load the questions
        into memory once and then read them from memory each epoch.
    max_token_length: ``int``, optional, (default: ``None``)
        If specified, we will truncate longer tokens to this length in the
        passages (but not in the questions).
    tokenizer : ``Tokenizer``, optional
        We'll use this tokenizer on questions and evidence passages, defaulting to
        ``WordTokenizer`` if none is provided.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Determines how both the question and the evidence passages are represented as arrays.  See
        :class:`TokenIndexer`.  Default is to have a single word ID for every token.
    """
    # Class static variable for storing the temp locations of untarred files,
    # so that if we instantiate two different dataset readers for the
    # same tarball we can reuse the temp file
    _temp_files: Dict[str, Tuple[pathlib.Path, int]] = {}

    def __init__(self,
                 triviaqa_path: str,
                 unfiltered_path: str = None,
                 data_format: str = "tar.gz",
                 paragraph_picker: str = None,
                 sample_first_iteration: bool = False,
                 keep_questions_in_memory: bool = True,
                 max_token_length: int = None,
                 max_paragraphs: int = 4,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        # Store original paths as strings
        self._triviaqa_path = triviaqa_path
        self._unfiltered_path = unfiltered_path

        self._sample_this_iteration = sample_first_iteration
        self._keep_questions_in_memory = keep_questions_in_memory
        self._paragraph_picker = paragraph_picker
        self._max_token_length = max_token_length
        self._max_paragraphs = max_paragraphs
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        # In-memory cache for questions.
        self._data: Dict[str, List[Question]] = {}

        if data_format not in ["preprocessed", "tar.gz", "unpackaged"]:
            raise ConfigurationError(f"unknown data_format {data_format}")

        self._preprocessed = (data_format == "preprocessed")

        if data_format == "tar.gz":
            # Need to untar, unless it's already been done.
            self._evidence_path = self._untar(self._triviaqa_path)
            if unfiltered_path:
                self._question_path = self._untar(self._unfiltered_path)
            else:
                self._question_path = self._evidence_path
        else:
            # Don't need to untar.
            self._evidence_path = pathlib.Path(self._triviaqa_path)
            if unfiltered_path:
                self._question_path = pathlib.Path(self._unfiltered_path)
            else:
                self._question_path = self._evidence_path

    def _untar(self, tar_path: str) -> pathlib.Path:
        """
        Return the path to the untarred version of tar_path,
        extracting the tarfile first if necessary.
        """
        if tar_path in self._temp_files:
            logger.info(f"{tar_path} has already been unzipped, reusing")
            untarred_path, ref_count = self._temp_files[tar_path]
            self._temp_files[tar_path] = untarred_path, ref_count + 1
        else:
            logger.info(f"un-tar-ing {tar_path}")
            untarred_path = pathlib.Path(tempfile.mkdtemp())
            with tarfile.open(tar_path) as tarball:
                tarball.extractall(untarred_path)
            self._temp_files[tar_path] = untarred_path, 1
        return untarred_path

    def _cleanup(self, tar_path: str) -> None:
        """
        Decrement the reference count for ``tar_path``
        and delete it if there are no more references to it.
        """
        if tar_path and tar_path in self._temp_files:
            untarred_path, ref_count = self._temp_files[tar_path]
            if ref_count > 1:
                self._temp_files[tar_path] = untarred_path, ref_count - 1
            else:
                shutil.rmtree(untarred_path)
                del self._temp_files[tar_path]

    def __del__(self):
        """
        Clean up temporary directories
        """
        self._cleanup(self._triviaqa_path)
        self._cleanup(self._unfiltered_path)  # will be a no-op if unfiltered_path is None

    @overrides
    def _read(self, file_path: str):
        if file_path in self._data:
            # Data is already in memory, so just use it.
            questions = self._data[file_path]
        elif self._preprocessed:
            # Data is preprocessed in JSONL format, so load it that way.
            questions_path = self._question_path / file_path
            logger.info(f"loading preprocessed questions from {questions_path}")
            with open(questions_path, 'r') as f:
                questions = [Question.from_json(json.loads(line)) for line in f]
        else:
            # Data from untarred original file.
            logger.info(f"preprocessing questions from {file_path}")
            questions_path = self._question_path / 'qa' / file_path

            # Require a paragraph with an answer only if this is the training
            # dataset reader.
            require_answer = self._paragraph_picker == 'triviaqa-web-train'
            questions = [question for question in process_triviaqa_questions(evidence_path=self._evidence_path,
                                                                             questions_path=questions_path,
                                                                             tokenizer=self._tokenizer,
                                                                             max_paragraphs=self._max_paragraphs,
                                                                             require_answer=require_answer,
                                                                             max_paragraph_length=400)]

        # Store questions in memory, unless we're not supposed to.
        if self._keep_questions_in_memory and file_path not in self._data:
            self._data[file_path] = questions

        for question in questions:
            paragraphs = question.paragraphs
            if self._sample_this_iteration and self._paragraph_picker == 'triviaqa-web-train':
                sample: List[int] = []
                # double sample the first one
                choices = [0] + [i for i in range(len(paragraphs.texts))]
                while not any(i in paragraphs.has_answers for i in sample):
                    sample = np.random.choice(choices, size=2)
                picked_paragraph_texts = [paragraphs.texts[i] for i in sample]
                picked_paragraph_tokens = [paragraphs.tokens[i] for i in sample]
                picked_paragraph_spans = [paragraphs.token_spans[i] for i in sample]
            else:
                picked_paragraph_texts = paragraphs.texts
                picked_paragraph_tokens = paragraphs.tokens
                picked_paragraph_spans = paragraphs.token_spans

            if not picked_paragraph_tokens:
                continue

            # Truncate tokens if necessary
            if self._max_token_length is not None:
                picked_paragraph_tokens = [[truncate_token(token, self._max_token_length) for token in tokens]
                                           for tokens in picked_paragraph_tokens]

            instance = util.make_multi_paragraph_reading_comprehension_instance(
                    question.tokens,
                    picked_paragraph_tokens,
                    self._token_indexers,
                    picked_paragraph_texts,
                    picked_paragraph_spans,
                    question.answer_texts)

            yield instance

        # sample_this_iteration is always True after the first iteration
        self._sample_this_iteration = True

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         paragraphs: List[str],
                         token_spans: List[List[Tuple[int, int]]] = None,
                         answer_texts: List[str] = None,
                         question_tokens: List[Token] = None,
                         paragraph_tokens: List[List[Token]] = None) -> Instance:
        # pylint: disable=arguments-differ
        if paragraph_tokens is None:
            paragraph_tokens = [self._tokenizer.tokenize(paragraph) for paragraph in paragraphs]

        if self._max_token_length is not None:
            paragraph_tokens = [[truncate_token(token, self._max_token_length) for token in tokens]
                                for tokens in paragraph_tokens]

        if token_spans is None:
            token_spans = [util.find_valid_answer_spans(paragraph_tokens_i, answer_texts)
                           for paragraph_tokens_i in paragraph_tokens]
        if question_tokens is None:
            question_tokens = self._tokenizer.tokenize(question_text)

        return util.make_multi_paragraph_reading_comprehension_instance(
                question_tokens,
                paragraph_tokens,
                self._token_indexers,
                paragraphs,
                token_spans,
                answer_texts)

    @classmethod
    def from_params(cls, params: Params) -> 'TriviaQaReader':
        triviaqa_path = params.pop('triviaqa_path')
        data_format = params.pop('data_format', 'tar.gz')
        paragraph_picker = params.pop('paragraph_picker', None)
        sample_first_iteration = params.pop_bool('sample_first_iteration', False)
        keep_questions_in_memory = params.pop_bool('keep_questions_in_memory', True)
        max_token_length = params.pop_int('max_token_length', None)
        max_paragraphs = params.pop_int('max_paragraphs', 4)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(triviaqa_path=triviaqa_path,
                   data_format=data_format,
                   sample_first_iteration=sample_first_iteration,
                   keep_questions_in_memory=keep_questions_in_memory,
                   paragraph_picker=paragraph_picker,
                   max_token_length=max_token_length,
                   max_paragraphs=max_paragraphs,
                   tokenizer=tokenizer,
                   token_indexers=token_indexers,
                   lazy=lazy)
