import json
import logging
import os
import tarfile
from typing import Dict, List, Tuple

from overrides import overrides

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
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._base_tarball_path = base_tarball_path
        self._unfiltered_tarball_path = unfiltered_tarball_path
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

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

            evidence_files: List[List[str]] = []  # contains lines from each evidence file
            if 'web' in file_path:
                for result in question_json['SearchResults']:
                    filename = result['Filename']
                    evidence_file = base_tarball.extractfile(os.path.join("evidence", "web", filename))
                    evidence_files.append([line.decode('utf-8') for line in evidence_file.readlines()])
            else:
                for result in question_json['EntityPages']:
                    filename = result['Filename']
                    evidence_file = base_tarball.extractfile(os.path.join("evidence", "wikipedia", filename))
                    evidence_files.append([line.decode('utf-8') for line in evidence_file.readlines()])

            answer_json = question_json['Answer']
            human_answers = [util.normalize_text(answer) for answer in answer_json.get('HumanAnswers', [])]
            answer_texts = answer_json['NormalizedAliases'] + human_answers
            for paragraph in self.pick_paragraphs(evidence_files, question_text, answer_texts):
                paragraph_tokens = self._tokenizer.tokenize(paragraph)
                token_spans = util.find_valid_answer_spans(paragraph_tokens, answer_texts)
                if not token_spans:
                    # For now, we'll just ignore instances that we can't find answer spans for.
                    # Maybe we can do something smarter here later, but this will do for now.
                    continue
                instance = self.text_to_instance(question_text,
                                                 paragraph,
                                                 token_spans,
                                                 answer_texts,
                                                 question_tokens,
                                                 paragraph_tokens)
                yield instance

    def pick_paragraphs(self,
                        evidence_files: List[List[str]],
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
        # pylint: disable=no-self-use,unused-argument
        paragraphs = []
        for evidence_file in evidence_files:
            whole_document = ' '.join(evidence_file)
            tokens = whole_document.split(' ')
            paragraph = ' '.join(tokens[:400])
            paragraphs.append(paragraph)
        return paragraphs

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
                                                        answer_texts)

    @classmethod
    def from_params(cls, params: Params) -> 'TriviaQaReader':
        base_tarball_path = params.pop('base_tarball_path')
        unfiltered_tarball_path = params.pop('unfiltered_tarball_path', None)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(base_tarball_path=base_tarball_path,
                   unfiltered_tarball_path=unfiltered_tarball_path,
                   tokenizer=tokenizer,
                   token_indexers=token_indexers,
                   lazy=lazy)
