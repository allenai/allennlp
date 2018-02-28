import json
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile
from typing import Dict, List, Tuple, Iterable, Iterator, NamedTuple, Set

from overrides import overrides
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np

from allennlp.common import Params, JsonDict
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

_PARAGRAPH_TOKEN = Token("@@PARAGRAPH@@")

NUM_QUESTIONS = 10

class MergedParagraphs(NamedTuple):
    texts: List[str]
    tokens: List[List[Token]]
    has_answers: Set[int] = None

@DatasetReader.register("triviaqa")
class TriviaQaReader(DatasetReader):
    """
    Reads the TriviaQA dataset into a ``Dataset`` containing ``Instances`` with three fields:
    ``question`` (a ``TextField``), ``paragraphs`` (a ``ListField[TextField]``),
    and ``spans`` (a ``ListField[SpanField]``). Each instance consists of one or more paragraphs
    from the same document, chosen in a manner specified by the ``paragraph_picker`` parameter.

    TriviaQA is split up into several JSON files defining the questions, and a lot of text files
    containing crawled web documents.  We read these from a gzipped tarball, to avoid having to
    have millions of individual files on a filesystem.

    Because we need to read both train and validation files from the same tarball, we take the
    tarball itself as a constructor parameter, and take the question file as the argument to
    ``read``.  This means that you should give the path to the tarball in the ``dataset_reader``
    parameters in your experiment configuration file, and something like ``"wikipedia-train.json"``
    for the ``train_data_path`` and ``validation_data_path``.

    Parameters
    ----------
    base_tarball_path : ``str``
        This is the path to the main ``tar.gz`` file you can download from the TriviaQA website,
        with directories ``evidence`` and ``qa``.
    unfiltered_tarball_path : ``str``, optional
        This is the path to the "unfiltered" TriviaQA data that you can download from the TriviaQA
        website, containing just question JSON files that point to evidence files in the base
        tarball.
    paragraph_picker: ``str``, optional, (default: ``None``)
        If specified, this indicates the scheme for sampling paragraphs
        for each question-document pair.
    cache_questions: ``bool``, optional, (default: ``True``)
        If ``True``, the JSON blobs representing questions will be stored
        in-memory between iterations. (It is on the order of 100s of MB
        and is somewhat slow to parse.)
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
    _temp_files: Dict[str, Tuple[str, int]] = {}

    def __init__(self,
                 base_tarball_path: str,
                 unfiltered_tarball_path: str = None,
                 paragraph_picker: str = None,
                 cache_questions: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        #self._unfiltered_tarball_path = unfiltered_tarball_path
        self._paragraph_picker = paragraph_picker
        self._cache_questions = cache_questions
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words="")

        self._questions: Dict[str, JsonDict] = {}

        # If we're really given a tarball, we need to extract it to a temp directory.

        if os.path.isdir(base_tarball_path):
            # Given a directory, so do nothing.
            logger.info(f"{base_tarball_path} is a directory, so no un-tar-ing to do.")
            self._base_tarball_path = pathlib.Path(base_tarball_path)
        else:
            # Otherwise, we need to untar triviaqa into a temp directory.
            if base_tarball_path in self._temp_files:
                # We've already untarred this, so just grab its location
                # and increment its reference count.
                tempdir, ref_count = self._temp_files[base_tarball_path]
                self._temp_files[base_tarball_path] = (tempdir, ref_count + 1)
                logger.info(f"{base_tarball_path} has already been un-tar-ed, will just reuse.")
            else:
                # Make a new tempdir, untar the dataset, and store the location.
                tempdir = tempfile.mkdtemp()
                logger.info(f"Un-tar-ing {base_tarball_path} to {tempdir}")
                with tarfile.open(base_tarball_path) as tarball:
                    tarball.extractall(tempdir)
                self._temp_files[base_tarball_path] = (tempdir, 1)

            self._base_tarball_path = pathlib.Path(tempdir)

    def __del__(self):
        """At cleanup time, delete a temp directory that we don't need anymore"""
        keys = [key for key in self._temp_files]

        for base_tarball_path in keys:
            tempfile, ref_count = self._temp_files[base_tarball_path]
            if tempfile == str(self._base_tarball_path):
                # Our path is a tempfile
                if ref_count == 1:
                    # And this was the only usage of it, so clean it up.
                    logger.info("cleaning up temp directory")
                    shutil.rmtree(tempfile)
                    del self._temp_files[base_tarball_path]
                else:
                    # Someone else is using it, so just decrement its reference count.
                    self._temp_files[base_tarball_path] = (tempfile, ref_count - 1)

    @overrides
    def _read(self, file_path: str):
        if 'web' in file_path:
            result_key, evidence_subdir = 'SearchResults', 'web'
        else:
            result_key, evidence_subdir = 'EntityPages', 'wikipedia'

        question_path = self._base_tarball_path / 'qa' / file_path

        if question_path in self._questions:
            data_json = self._questions[question_path]
        else:
            logger.info("Loading question file from tarball")
            with open(question_path, 'r') as f:
                data_json = json.loads(f.read())
            self._questions[question_path] = data_json
            logger.info("Loaded questions into memory")

        questions_data = data_json['Data'][:NUM_QUESTIONS]
        num_evidence_files = sum(len(question_json[result_key])
                                 for question_json in questions_data)
        logger.info(f"reading {num_evidence_files} evidence files")

        for question_json in questions_data[:NUM_QUESTIONS]:
            question_text = question_json['Question']
            question_tokens = self._tokenizer.tokenize(question_text)

            answer_json = question_json['Answer']
            human_answers = [util.normalize_text(answer) for answer in answer_json.get('HumanAnswers', [])]
            answer_texts = answer_json['NormalizedAliases'] + human_answers

            evidence_files = [result['Filename'] for result in question_json[result_key]]

            for evidence_filename in evidence_files:
                evidence_path = self._base_tarball_path / 'evidence' / evidence_subdir / evidence_filename
                with open(evidence_path, 'r') as evidence_file:
                    paragraphs = [line for line in evidence_file.readlines()]
                merged_paragraphs = self.merge_and_sort(paragraphs, question_text, answer_texts)

                if not merged_paragraphs:
                    # TODO(joelgrus) what here
                    continue

                # Pick paragraphs
                if self._paragraph_picker == 'triviaqa-web-train':
                    sample: List[int] = []
                    # Sample until we get at least one paragraph with an answer
                    num_texts = len(merged_paragraphs.texts)

                    if merged_paragraphs.has_answers:
                        while not any(i in merged_paragraphs.has_answers for i in sample):
                            sample = np.random.choice(np.arange(num_texts), size=2)
                        picked_paragraphs = [merged_paragraphs.texts[i] for i in sample]
                        picked_paragraph_tokens = [merged_paragraphs.tokens[i] for i in sample]
                        instance = self.text_to_instance(question_text=question_text,
                                                         question_tokens=question_tokens,
                                                         paragraphs=picked_paragraphs,
                                                         paragraph_tokens=picked_paragraph_tokens,
                                                         answer_texts=answer_texts)
                    else:
                        instance = None
                else:
                    instance = self.text_to_instance(question_text=question_text,
                                                     question_tokens=question_tokens,
                                                     paragraphs=merged_paragraphs.texts,
                                                     paragraph_tokens=merged_paragraphs.tokens,
                                                     answer_texts=answer_texts)

                if instance is not None:
                    yield instance


    def document_tfidf(self, paragraphs: List[str], question: str) -> np.ndarray:
        try:
            # (num_paragraphs, num_features)
            para_features = self._tfidf.fit_transform(paragraphs)
            # (1, num_features)
            q_features = self._tfidf.transform([question])
        except ValueError:
            # (num_paragraphs,)
            return np.array([0.0] * len(paragraphs))
        # pairwise_distances is (1, num_paragraphs), after ravel is (num_paragraphs,)
        dists = pairwise_distances(q_features, para_features, "cosine").ravel()
        return dists


    def merge_and_sort(self,
                       paragraphs: List[str],
                       question: str = None,
                       answer_texts: List[str] = None,
                       max_size: int = 400) -> MergedParagraphs:

        # current_paragraph, current_tokens = "", []
        # merged_paragraphs: List[str] = []

        # for paragraph in paragraphs:
        #     paragraph_tokens = self._tokenizer.tokenize(paragraph)

        #     if len(current_tokens) + len(paragraph_tokens) + 1 > max_size:
        #         # Too big, so add to output
        #         merged_paragraphs.append(current_paragraph)
        #         current_tokens = []
        #         current_paragraph = paragraph
        #     else:
        #         # Keep appending
        #         current_paragraph += f" {_PARAGRAPH_TOKEN.text} " + paragraph
        #         current_tokens.extend(paragraph_tokens)
        #         current_tokens.append(Token(_PARAGRAPH_TOKEN))

        # if current_paragraph:
        #     merged_paragraphs.append(current_paragraph)


        tokens = []
        for paragraph in paragraphs:
            tokens.extend(token.text for token in self._tokenizer.tokenize(paragraph))
            tokens.append(_PARAGRAPH_TOKEN.text)

        # Get rid of trailing paragraph token
        tokens = tokens[:-1]

        merged_paragraphs = [' '.join(paragraph_tokens) for paragraph_tokens in lazy_groups_of(iter(tokens), max_size)]
        merged_paragraph_tokens = [self._tokenizer.tokenize(paragraph) for paragraph in merged_paragraphs]

        # If we're training, we can prune the paragraphs down:
        if self._paragraph_picker == "triviaqa-web-train":
            # Take the top four paragraphs ranked by tf-idf score and sample two from them.
            # Sample the highest-ranked paragraph that contains an answer twice as often.
            # Require at least one of the paragraphs to contain an answer span.

            # Sort the paragraphs by their tfidf score with the question.
            scores = self.document_tfidf(merged_paragraphs, question)
            # Get the ranked indexes.
            ranks = [i for i, _ in sorted(enumerate(scores), key=lambda pair: pair[1])]

            # Find the indexes of paragraphs that have answers.
            has_answers = [i for i in ranks
                           if util.find_valid_answer_spans(merged_paragraph_tokens[i], answer_texts)]

            if has_answers:
                # Want to sample the highest rank answer twice as often.
                first_answer = has_answers[0]
                if first_answer < 4:
                    choices = [first_answer] + ranks[:4]
                else:
                    choices = [first_answer, first_answer] + ranks[:3]

                return MergedParagraphs(texts=[merged_paragraphs[i] for i in choices],
                                        tokens=[merged_paragraph_tokens[i] for i in choices],
                                        has_answers={i for i, choice in enumerate(choices) if choice in has_answers})

            else:
                # No paragraphs that include an answer!
                # TODO(joelgrus) should we do something else here?
                return None
        else:
            # Not sampling
            has_answers = [i for i, mpt in enumerate(merged_paragraph_tokens)
                           if util.find_valid_answer_spans(mpt, answer_texts)]
            if not has_answers:
                return None
            else:
                return MergedParagraphs(texts=merged_paragraphs,
                                        tokens=merged_paragraph_tokens,
                                        has_answers=has_answers)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         paragraphs: List[str],
                         token_spans: List[Tuple[int, int]] = None,
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
        base_tarball_path = params.pop('base_tarball_path')
        unfiltered_tarball_path = params.pop('unfiltered_tarball_path', None)
        paragraph_picker = params.pop('paragraph_picker', None)
        cache_questions = params.pop_bool('cache_questions', True)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(base_tarball_path=base_tarball_path,
                   unfiltered_tarball_path=unfiltered_tarball_path,
                   paragraph_picker=paragraph_picker,
                   cache_questions=cache_questions,
                   tokenizer=tokenizer,
                   token_indexers=token_indexers,
                   lazy=lazy)
