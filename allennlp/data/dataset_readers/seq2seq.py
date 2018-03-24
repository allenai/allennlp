from typing import Dict
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

START_SYMBOL = "@@START@@"
END_SYMBOL = "@@END@@"

@DatasetReader.register("seq2seq")
class Seq2SeqDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    source_max_sequence_length : int, (optional, default=0)
        Maximum source sequence length (excluding `START_SYMBOL` and `END_SYMBOL`).
        Examples with source sequence length exceeding this value are discarded.
        0 indicates length is unlimited. Value must be greater than or equal to 0.
    source_truncate_sequence_length : int, (optional, default=0)
        Source sequences longer than this value (excluding `START_SYMBOL` and
        `END_SYMBOL`) will be truncated to this value (cutting starting from the end).
        0 indicates length is unlimited. Value must be greater than or equal to 0.
    target_max_sequence_length : int, (optional, default=0)
        Maximum target sequence length (excluding `START_SYMBOL` and `END_SYMBOL`).
        Examples with target sequence length exceeding this value are discarded.
        0 indicates length is unlimited. Value must be greater than or equal to 0.
    target_truncate_sequence_length : int, (optional, default=0)
        Target sequences longer than this value (excluding `START_SYMBOL` and
        `END_SYMBOL`) will be truncated to this value (cutting starting from the end).
        0 indicates length is unlimited. Value must be greater than or equal to 0.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 source_max_sequence_length: int = 0,
                 source_truncate_sequence_length: int = 0,
                 target_max_sequence_length: int = 0,
                 target_truncate_sequence_length: int = 0,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token

        if source_max_sequence_length < 0:
            raise ConfigurationError(
                "source_max_sequence_length is {}, but must "
                "be greater than or equal to 0".format(source_max_sequence_length))
        self._source_max_sequence_length = source_max_sequence_length
        self._filter_source = (source_max_sequence_length != 0)

        if source_truncate_sequence_length < 0:
            raise ConfigurationError(
                "source_truncate_sequence_length is {}, but must "
                "be greater than or equal to 0".format(source_truncate_sequence_length))
        self._source_truncate_sequence_length = source_truncate_sequence_length
        self._truncate_source = (source_truncate_sequence_length != 0)

        if target_max_sequence_length < 0:
            raise ConfigurationError(
                "target_max_sequence_length is {}, but must "
                "be greater than or equal to 0".format(target_max_sequence_length))
        self._target_max_sequence_length = target_max_sequence_length
        self._filter_target = (target_max_sequence_length != 0)

        if target_truncate_sequence_length < 0:
            raise ConfigurationError(
                "target_truncate_sequence_length is {}, but must "
                "be greater than or equal to 0".format(target_truncate_sequence_length))
        self._target_truncate_sequence_length = target_truncate_sequence_length
        self._truncate_target = (target_truncate_sequence_length != 0)

    @overrides
    def _read(self, file_path):
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                source_sequence, target_sequence = line_parts
                instance = self.text_to_instance(source_sequence, target_sequence)
                if instance is None:
                    continue
                yield instance

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if len(tokenized_source) > self._source_max_sequence_length and self._filter_source:
            return None
        if len(tokenized_source) > self._source_truncate_sequence_length and self._truncate_source:
            tokenized_source = tokenized_source[:self._source_truncate_sequence_length]

        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            if len(tokenized_target) > self._target_max_sequence_length and self._filter_target:
                return None
            if (len(tokenized_target) > self._target_truncate_sequence_length and
                    self._truncate_target):
                tokenized_target = tokenized_target[:self._target_truncate_sequence_length]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field})

    @classmethod
    def from_params(cls, params: Params) -> 'Seq2SeqDatasetReader':
        source_tokenizer_type = params.pop('source_tokenizer', None)
        source_tokenizer = None if source_tokenizer_type is None else Tokenizer.from_params(source_tokenizer_type)
        target_tokenizer_type = params.pop('target_tokenizer', None)
        target_tokenizer = None if target_tokenizer_type is None else Tokenizer.from_params(target_tokenizer_type)
        source_indexers_type = params.pop('source_token_indexers', None)
        source_add_start_token = params.pop_bool('source_add_start_token', True)
        source_max_sequence_length = params.pop_int("source_max_sequence_length", 0)
        source_truncate_sequence_length = params.pop_int("source_truncate_sequence_length", 0)
        target_max_sequence_length = params.pop_int("target_max_sequence_length", 0)
        target_truncate_sequence_length = params.pop_int("target_truncate_sequence_length", 0)

        if source_indexers_type is None:
            source_token_indexers = None
        else:
            source_token_indexers = TokenIndexer.dict_from_params(source_indexers_type)
        target_indexers_type = params.pop('target_token_indexers', None)
        if target_indexers_type is None:
            target_token_indexers = None
        else:
            target_token_indexers = TokenIndexer.dict_from_params(target_indexers_type)
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return Seq2SeqDatasetReader(source_tokenizer, target_tokenizer,
                                    source_token_indexers, target_token_indexers,
                                    source_add_start_token,
                                    source_max_sequence_length,
                                    source_truncate_sequence_length,
                                    target_max_sequence_length,
                                    target_truncate_sequence_length,
                                    lazy)
