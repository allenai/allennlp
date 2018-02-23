import json
import logging
import os
import tarfile
from typing import Dict, List, Tuple, Iterable, Iterator, NamedTuple, Set

from overrides import overrides
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np

from allennlp.common import Params, JsonDict
from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MAX_INSTANCES = 1000

_PARAGRAPH_TOKEN = Token("@@PARAGRAPH@@")

class MergedParagraphs(NamedTuple):
    texts: List[str]
    tokens: List[List[Token]]
    has_answers: Set[int] = None

class Question(NamedTuple):
    text: str
    tokens: List[Token]
    answer_texts: List[str]
    evidence_files: List[str]

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
        self._merged_paragraphs: Dict[str, MergedParagraphs] = {}
        self._base_tarball_path = base_tarball_path


    @overrides
    def _read(self, file_path: str):
        if 'web' in file_path:
            result_key, evidence_subdir = 'SearchResults', 'web'
        else:
            result_key, evidence_subdir = 'EntityPages', 'wikipedia'

        questions = self._questions.get(file_path)

        if questions is None:
            with tarfile.open(cached_path(self._base_tarball_path), 'r') as base_tarfile:
                logger.info("Opening base tarball file at %s", self._base_tarball_path)

                logger.info("Loading question file from tarball")
                path = os.path.join('qa', file_path)
                data_json = json.loads(base_tarfile.extractfile(path).read().decode('utf-8'))
                logger.info("Loaded questions into memory")

                # Now we need an index from archive name to question:
                index = {}

                questions: List[Question] = []

                questions_data = data_json['Data']
                for question_json in Tqdm.tqdm(questions_data, total=len(questions_data), desc='indexing questions'):
                    question_text = question_json['Question']
                    question_tokens = self._tokenizer.tokenize(question_text)

                    answer_json = question_json['Answer']
                    human_answers = [util.normalize_text(answer) for answer in answer_json.get('HumanAnswers', [])]
                    answer_texts = answer_json['NormalizedAliases'] + human_answers

                    evidence_files = [result['Filename'] for result in question_json[result_key]]

                    question = Question(text=question_text,
                                        tokens=question_tokens,
                                        answer_texts=answer_texts,
                                        evidence_files=evidence_files)

                    questions.append(question)

                    for evidence_file in evidence_files:
                        index[evidence_file] = question

                self._questions[file_path] = questions

                # Now load the evidence files:
                logger.info("loading evidence files")
                infos = (tar_info
                         for tar_info in base_tarfile
                         if tar_info.name.startswith(f"evidence/{evidence_subdir}")
                         and tar_info.isfile()
                         and '/'.join(tar_info.name.split("/")[-2:]) in index)

                for tar_info in Tqdm.tqdm(infos, desc='relevant evidence files'):
                    filename = '/'.join(tar_info.name.split("/")[-2:])
                    evidence_file = base_tarfile.extractfile(tar_info)
                    question = index[filename]
                    paragraphs = [line.decode('utf-8') for line in evidence_file.readlines()]
                    merged_paragraphs = self.merge_and_sort(paragraphs, question.text, question.answer_texts)
                    self._merged_paragraphs[filename] = merged_paragraphs


        # Now we can iterate more pleasantly
        logger.info("Reading the dataset")
        questions = self._questions[file_path]

        for question in questions:
            for evidence_file in question.evidence_files:
                merged_paragraphs = self._merged_paragraphs[evidence_file]

                if merged_paragraphs is None:
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
                        instance = self.text_to_instance(question_text=question.text,
                                                         question_tokens=question.tokens,
                                                         paragraphs=picked_paragraphs,
                                                         paragraph_tokens=picked_paragraph_tokens,
                                                         answer_texts=question.answer_texts)
                    else:
                        instance = None
                else:
                    instance = self.text_to_instance(question_text=question.text,
                                                     question_tokens=question.tokens,
                                                     paragraphs=merged_paragraphs.texts,
                                                     paragraph_tokens=merged_paragraphs.tokens,
                                                     answer_texts=question.answer_texts)

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

        current_paragraph, current_tokens = "", []

        merged_paragraphs: List[str] = []

        for paragraph in paragraphs:
            paragraph_tokens = self._tokenizer.tokenize(paragraph)

            if len(current_tokens) + len(paragraph_tokens) + 1 > max_size:
                # Too big, so add to output
                merged_paragraphs.append(current_paragraph)
                current_paragraph = paragraph
            else:
                # Keep appending
                current_paragraph += f" {_PARAGRAPH_TOKEN.text} " + paragraph

        if current_paragraph:
            merged_paragraphs.append(current_paragraph)

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
            MergedParagraphs(texts=merged_paragraphs,
                             tokens=merged_paragraph_tokens)




    # def pick_paragraphs(self,
    #                     paragraphs: List[str],
    #                     question: str = None,
    #                     answer_texts: List[str] = None) -> List[str]:
    #     """
    #     Given a list of evidence documents, return a list of paragraphs to use as training
    #     examples.  Each paragraph returned will be made into one training example.

    #     To aid in picking the best paragraph, you can also optionally pass the question text or the
    #     answer strings.  Note, though, that if you actually use the answer strings for picking the
    #     paragraph on the dev or test sets, that's likely cheating, depending on how you've defined
    #     the task.
    #     """
    #     paragraphs, paragraph_tokens = zip(*self.merged_paragraphs(paragraphs))

    #     if self._paragraph_picker == "triviaqa-web-train":
    #         # Take the top four paragraphs ranked by tf-idf score and sample two from them.
    #         # Sample the highest-ranked paragraph that contains an answer twice as often.
    #         # Require at least one of the paragraphs to contain an answer span.

    #         # Sort the paragraphs by their tfidf score with the question.
    #         scores = self.document_tfidf(paragraphs, question)
    #         # Get the ranked indexes.
    #         ranks = [i for i, _ in sorted(enumerate(scores), key=lambda pair: pair[1])]

    #         # Find the indexes of paragraphs that have answers.
    #         has_answers = [i for i in ranks
    #                        if util.find_valid_answer_spans(paragraph_tokens[i], answer_texts)]

    #         if has_answers:
    #             # Want to sample the highest rank answer twice as often.
    #             first_answer = has_answers[0]
    #             if first_answer < 4:
    #                 choices = [first_answer] + ranks[:4]
    #             else:
    #                 choices = [first_answer, first_answer] + ranks[:3]

    #             sample: Iterable[int] = []
    #             # Sample until we get at least one paragraph with an answer
    #             while not any(i in has_answers for i in sample):
    #                 sample = np.random.choice(choices, size=2)
    #             picked_paragraphs = [paragraphs[i] for i in sample]
    #         else:
    #             # No paragraphs that include an answer!
    #             # TODO(joelgrus) should we do something else here?
    #             picked_paragraphs = []
    #     else:
    #         picked_paragraphs = paragraphs

    #     return picked_paragraphs

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
