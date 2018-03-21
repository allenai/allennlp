# pylint: disable=no-member
import json
import logging
import pathlib
import shutil
import tarfile
import tempfile
from typing import Dict, List, Tuple, Iterator, NamedTuple

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

class MergedParagraphs(NamedTuple):
    """
    A typical TriviaQA question has multiple files with answers, each consisting
    of many small paragraphs. We preprocess these by merging them into fewer
    paragraphs of larger size.
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

    def truncate_passage_tokens(self, max_len: int = None) -> None:
        if max_len is None:
            return

        # This is hacky, but ``MergedParagraphs`` is immutable, so we can't change its ``.tokens``.
        # However, ``.tokens`` is a list of lists, so we can change the inner lists. Hacky.
        num_paragraphs = len(self.paragraphs)
        for i in range(num_paragraphs):
            self.paragraphs.tokens[i] = [truncate_token(token, max_len) for token in self.paragraphs.tokens[i]]

    def to_json(self) -> JsonDict:
        return {
                "id": self.id,
                "text": self.text,
                "tokens": [json_to_token(token) for token in self.tokens],
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


def _document_tfidf(tfidf: TfidfVectorizer, paragraphs: List[str], question: str) -> np.ndarray:
    """
    Given a question and some supporting paragraphs, return the array of tfidf
    distances from each paragraph to the question.
    """
    try:
        # (num_paragraphs, num_features)
        para_features = tfidf.fit_transform(paragraphs)
        # (1, num_features)
        q_features = tfidf.transform([question])
    except ValueError:
        # (num_paragraphs,)
        return np.array([0.0] * len(paragraphs))

    # pairwise_distances is (1, num_paragraphs); after ravel it's (num_paragraphs,)
    return pairwise_distances(q_features, para_features, "cosine").ravel()

def _merge_and_sort(paragraphs: List[str],
                    question: str,
                    answer_texts: List[str],
                    tokenizer: Tokenizer,
                    topn: int = 4,
                    max_paragraph_length: int = 400) -> MergedParagraphs:
    tfidf = TfidfVectorizer(strip_accents="unicode", stop_words="")

    tokens: List[str] = []
    for paragraph in paragraphs:
        tokens.extend(token.text for token in tokenizer.tokenize(paragraph))
        tokens.append(_PARAGRAPH_TOKEN)

    # Get rid of trailing paragraph token
    tokens = tokens[:-1]

    merged_paragraphs = [' '.join(paragraph_tokens)
                         for paragraph_tokens in lazy_groups_of(iter(tokens), max_paragraph_length)]
    merged_paragraph_tokens = [tokenizer.tokenize(paragraph) for paragraph in merged_paragraphs]

    # Sort the paragraphs by their tfidf score with the question.
    scores = _document_tfidf(tfidf, merged_paragraphs, question)
    # Get the ranked indexes.
    ranks = [i for i, _ in sorted(enumerate(scores), key=lambda pair: pair[1])]

    # Find the indexes of paragraphs that have answers.
    has_answers = [i for i in ranks
                   if util.find_valid_answer_spans(merged_paragraph_tokens[i], answer_texts)]

    if not has_answers:
        return None

    first_answer = has_answers[0]
    # Want first_answer to be the first paragraph, and then take the most highly ranked
    # other topn - 1
    choices = [first_answer] + [i for i in ranks if i != first_answer][:(topn - 1)]

    merged_texts = [merged_paragraphs[i] for i in choices]
    merged_tokens = [merged_paragraph_tokens[i] for i in choices]
    merged_token_spans = [util.find_valid_answer_spans(tokens_i, answer_texts)
                          for tokens_i in merged_tokens]

    return MergedParagraphs(texts=merged_texts,
                            tokens=merged_tokens,
                            token_spans=merged_token_spans,
                            has_answers=[i for i, choice in enumerate(choices) if choice in has_answers])


def preprocess(base_path: pathlib.Path,
               questions_file: str,
               tokenizer: Tokenizer,
               topn: int = 4,
               max_paragraph_length: int = 400) -> Iterator[Question]:
    """
    Preprocesses one of the questions files in the (untarred) TriviaQA dataset
    into ``Question`` objects. For each question in the specified JSON file,
    we load all of the evidence files, merge the paragraphs to get new paragraphs
    of size ``max_paragraph_length``, sort them by tfidf distance to the question,
    and take the ``topn`` closest paragraphs.
    """
    result_key, evidence_subdir = 'SearchResults', 'web'

    questions_path = base_path / 'qa' / questions_file

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
            evidence_path = base_path / 'evidence' / evidence_subdir / evidence_file
            with open(evidence_path, 'r') as evidence_file:
                paragraphs.extend(evidence_file.readlines())

        merged_paragraphs = _merge_and_sort(paragraphs=paragraphs,
                                            question=question_text,
                                            answer_texts=answer_texts,
                                            tokenizer=tokenizer,
                                            topn=topn,
                                            max_paragraph_length=max_paragraph_length)

        if not merged_paragraphs:
            logger.warning(f"found no paragraphs with answers for {question_id}, skipping")
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
    data_format : ``str``, optional (default: "tar.gz")
        Valid values are ``"tar.gz"``, for the single file you can download from the TriviaQA website,
        ``"unpackaged"``, for if you've extracted the tar file into a directory, or
        ``"preprocessed"``, for if you've already preprocessed it into serialized ``Question``s.
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
    _temp_files: Dict[pathlib.Path, Tuple[str, int]] = {}

    def __init__(self,
                 triviaqa_path: str,
                 data_format: str = "tar.gz",
                 paragraph_picker: str = None,
                 sample_first_iteration: bool = False,
                 max_token_length: int = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._triviaqa_path = triviaqa_path
        self._sample_this_iteration = sample_first_iteration
        self._paragraph_picker = paragraph_picker
        self._max_token_length = max_token_length
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self._data: Dict[str, List[Question]] = {}

        if data_format not in ["preprocessed", "tar.gz", "unpackaged"]:
            raise ConfigurationError(f"unknown data_format {data_format}")

        self._preprocessed = (data_format == "preprocessed")
        self._path = pathlib.Path(triviaqa_path)

        if data_format == "tar.gz":
            # need to untar, unless already untarred
            if self._path in self._temp_files:
                logger.info("already un-tar-ed, reusing")
                directory, ref_count = self._temp_files[self._path]
                self._temp_files[self._path] = (directory, ref_count + 1)
            else:
                logger.info("un-tar-ing")
                directory = tempfile.mkdtemp()
                with tarfile.open(triviaqa_path) as tarball:
                    tarball.extractall(directory)
                self._temp_files[self._path] = (directory, 1)

            self._path = pathlib.Path(directory)

    def __del__(self):
        """
        Clean up temporary directories
        """
        if self._triviaqa_path in self._temp_files:
            directory, ref_count = self._temp_files[self._triviaqa_path]
            if ref_count > 1:
                self._temp_files[self._triviaqa_path] = (directory, ref_count - 1)
            else:
                shutil.rmtree(directory)

    def _truncate(self, questions: List[Question]) -> List[Question]:
        for question in questions:
            question.truncate_passage_tokens(max_len=self._max_token_length)
        return questions

    @overrides
    def _read(self, file_path: str):

        if file_path in self._data:
            questions = self._data[file_path]
        elif self._preprocessed:
            question_path = self._path / file_path
            logger.info(f"loading preprocessed questions from {question_path}")
            with open(question_path, 'r') as f:
                questions = [Question.from_json(json.loads(line)) for line in f]
            self._data[file_path] = self._truncate(questions)
        else:
            logger.info(f"preprocessing questions from {file_path}")
            questions = [question for question in preprocess(base_path=self._path,
                                                             questions_file=file_path,
                                                             tokenizer=self._tokenizer,
                                                             topn=4,
                                                             max_paragraph_length=400)]
            self._data[file_path] = self._truncate(questions)

        logger.info("sample this iteration %s", self._sample_this_iteration)

        for question in questions:
            paragraphs = question.paragraphs
            # sample:
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

            instance = util.make_multi_paragraph_reading_comprehension_instance(
                    question.tokens,
                    picked_paragraph_tokens,
                    self._token_indexers,
                    picked_paragraph_texts,
                    picked_paragraph_spans,
                    question.answer_texts)

            yield instance

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
        max_token_length = params.pop_int('max_token_length', None)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(triviaqa_path=triviaqa_path,
                   data_format=data_format,
                   sample_first_iteration=sample_first_iteration,
                   paragraph_picker=paragraph_picker,
                   max_token_length=max_token_length,
                   tokenizer=tokenizer,
                   token_indexers=token_indexers,
                   lazy=lazy)
