from typing import Dict, Iterable, Union, Optional, List
import logging
import math

from overrides import overrides

from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("simple_language_modeling")
class SimpleLanguageModelingDatasetReader(DatasetReader):
    """
    Reads sentences, one per line, for language modeling. This does not handle arbitrarily formatted
    text with sentences spanning multiple lines.

    # Parameters

    tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the input sentences into words or other kinds of tokens. Defaults
        to `SpacyTokenizer()`.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input token representations. Defaults to
        `{"tokens": SingleIdTokenIndexer()}`.
    max_sequence_length : `int`, optional
        If specified, sentences with more than this number of tokens will be dropped.
    start_tokens : `List[str]`, optional (default=`None`)
        These are prepended to the tokens provided to the `TextField`.
    end_tokens : `List[str]`, optional (default=`None`)
        These are appended to the tokens provided to the `TextField`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_sequence_length: int = None,
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        **kwargs,
    ) -> None:
        if "lazy" not in kwargs:
            # We typically want language modeling data to be read lazily.
            kwargs["lazy"] = True
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if max_sequence_length is not None:
            self._max_sequence_length: Union[float, Optional[int]] = max_sequence_length
        else:
            self._max_sequence_length = math.inf

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]

        logger.info("Creating SimpleLanguageModelingDatasetReader")
        logger.info("max_sequence_length=%s", max_sequence_length)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentence: str,
    ) -> Instance:

        tokenized = self._tokenizer.tokenize(sentence)
        tokenized_with_ends = []
        tokenized_with_ends.extend(self._start_tokens)
        tokenized_with_ends.extend(tokenized)
        tokenized_with_ends.extend(self._end_tokens)
        return_instance = Instance({"source": TextField(tokenized_with_ends, self._token_indexers)})
        return return_instance

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        logger.info("Loading data from %s", file_path)
        dropped_instances = 0

        with open(file_path) as file:
            for sentence in file:
                instance = self.text_to_instance(sentence)
                if instance.fields["source"].sequence_length() <= self._max_sequence_length:
                    yield instance
                else:
                    dropped_instances += 1
        if not dropped_instances:
            logger.info(f"No instances dropped from {file_path}.")
        else:
            logger.warning(f"Dropped {dropped_instances} instances from {file_path}.")
