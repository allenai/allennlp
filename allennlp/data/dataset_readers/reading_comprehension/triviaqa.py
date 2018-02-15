import json
import logging
import os
import tarfile
from typing import Dict, List, Tuple, Iterable

from overrides import overrides
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("triviaqa")
class TriviaQaReader(DatasetReader):
    """
    Reads the TriviaQA dataset into a ``Dataset`` containing ``Instances`` with four fields:
    ``question`` (a ``TextField``), ``passage`` (another ``TextField``), ``span_start``, and
    ``span_end`` (both ``IndexFields``).

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
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._base_tarball_path = base_tarball_path
        self._unfiltered_tarball_path = unfiltered_tarball_path
        self._paragraph_picker = paragraph_picker
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words="")

    @overrides
    def _read(self, file_path: str):
        logger.info("Opening base tarball file at %s", self._base_tarball_path)
        base_tarball = tarfile.open(cached_path(self._base_tarball_path), 'r')
        if 'unfiltered' in file_path:
            logger.info("Opening unfiltered tarball file at %s", self._unfiltered_tarball_path)
            unfiltered_tarball = tarfile.open(cached_path(self._unfiltered_tarball_path), 'r')
            logger.info("Loading question file from tarball")
            data_json = json.loads(unfiltered_tarball.extractfile(file_path).read().decode('utf-8'))
        else:
            logger.info("Loading question file from tarball")
            path = os.path.join('qa', file_path)
            data_json = json.loads(base_tarball.extractfile(path).read().decode('utf-8'))

        logger.info("Reading the dataset")
        for question_json in data_json['Data']:
            question_text = question_json['Question']
            question_tokens = self._tokenizer.tokenize(question_text)

            answer_json = question_json['Answer']
            human_answers = [util.normalize_text(answer) for answer in answer_json.get('HumanAnswers', [])]
            answer_texts = answer_json['NormalizedAliases'] + human_answers

            if 'web' in file_path:
                result_key, evidence_subdir = 'SearchResults', 'web'
            else:
                result_key, evidence_subdir = 'EntityPages', 'wikipedia'

            for result in question_json[result_key]:
                filename = result['Filename']
                evidence_file = base_tarball.extractfile(os.path.join("evidence", evidence_subdir, filename))
                paragraphs = [line.decode('utf-8') for line in evidence_file]

                for paragraph in self.pick_paragraphs(paragraphs, question_text, answer_texts):
                    paragraph_tokens = self._tokenizer.tokenize(paragraph)
                    token_spans = util.find_valid_answer_spans(paragraph_tokens, answer_texts)
                    instance = self.text_to_instance(question_text,
                                                     paragraph,
                                                     token_spans,
                                                     answer_texts,
                                                     question_tokens,
                                                     paragraph_tokens)

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

    def pick_paragraphs(self,
                        paragraphs: List[str],
                        question: str = None,
                        answer_texts: List[str] = None) -> List[str]:
        """
        Given a list of evidence documents, return a list of paragraphs to use as training
        examples.  Each paragraph returned will be made into one training example.

        To aid in picking the best paragraph, you can also optionally pass the question text or the
        answer strings.  Note, though, that if you actually use the answer strings for picking the
        paragraph on the dev or test sets, that's likely cheating, depending on how you've defined
        the task.
        """
        # pylint: disable=unused-argument
        picked_paragraphs = []

        if self._paragraph_picker == "triviaqa-web-train":
            # Take the top four paragraphs ranked by tf-idf score
            # Then sample two from them
            # Sample the highest-ranked paragraph that contains an answer twice as often
            # require at least one of the paragraphs to contain an answer span
            scores = self.document_tfidf(paragraphs, question)
            ranked = [paragraph for score, paragraph in sorted(zip(scores, paragraphs))][:4]

            has_answers = [i for i, paragraph in enumerate(ranked)
                           if util.find_valid_answer_spans(self._tokenizer.tokenize(paragraph), answer_texts)]

            if has_answers:
                sample: Iterable[int] = []
                # Sample until we get at least one paragraph with an answer
                while not any(i in has_answers for i in sample):
                    # Sample the highest ranked paragraph that contains an answer twice as often.
                    sample = np.random.choice([0, 1, 2, 3, has_answers[0]], size=2)
                picked_paragraphs.extend(ranked[i] for i in sample)
            else:
                # TODO(joelgrus) should we do something else here?
                pass
        else:
            # This is the ``None`` case
            whole_document = ' '.join(paragraphs)
            tokens = whole_document.split(' ')
            paragraph = ' '.join(tokens[:400])
            picked_paragraphs.append(paragraph)


        return picked_paragraphs

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         token_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         question_tokens: List[Token] = None,
                         passage_tokens: List[Token] = None) -> Instance:
        # pylint: disable=arguments-differ
        if not question_tokens:
            question_tokens = self._tokenizer.tokenize(question_text)
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        return util.make_reading_comprehension_instance(question_tokens,
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts,
                                                        pick_most_common_answer=False)

    @classmethod
    def from_params(cls, params: Params) -> 'TriviaQaReader':
        base_tarball_path = params.pop('base_tarball_path')
        unfiltered_tarball_path = params.pop('unfiltered_tarball_path', None)
        paragraph_picker = params.pop('paragraph_picker', None)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(base_tarball_path=base_tarball_path,
                   unfiltered_tarball_path=unfiltered_tarball_path,
                   paragraph_picker=paragraph_picker,
                   tokenizer=tokenizer,
                   token_indexers=token_indexers,
                   lazy=lazy)
