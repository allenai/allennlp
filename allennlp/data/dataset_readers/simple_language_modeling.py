from typing import Dict, Iterable, Union, Optional
import logging
import math

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("simple_language_modeling")
class SimpleLanguageModelingDatasetReader(DatasetReader):
    """
    Reads sentences, one per line, for language modeling. This does not handle arbitrarily formatted
    text with sentences spanning multiple lines.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sentences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    max_sequence_length: ``int``, optional
        If specified, sentences with more than this number of tokens will be dropped.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None) -> None:
        super().__init__(True)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if max_sequence_length is not None:
            self._max_sequence_length: Union[float, Optional[int]] = max_sequence_length
        else:
            self._max_sequence_length = math.inf

        logger.info("Creating SimpleLanguageModelingDatasetReader")
        logger.info("max_sequence_length=%s", max_sequence_length)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized = self._tokenizer.tokenize(sentence)
        return_instance = Instance({
                'source': TextField(tokenized, self._token_indexers),
        })
        return return_instance

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # pylint: disable=arguments-differ
        logger.info('Loading data from %s', file_path)

        with open(file_path) as file:
            for sentence in file:
                instance = self.text_to_instance(sentence)
                if instance.fields['source'].sequence_length() <= self._max_sequence_length:
                    yield instance
