from typing import Dict
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

START_SYMBOL = "@@START@@"
END_SYMBOL = "@@END@@"

@DatasetReader.register("encoder_decoder")
class EncoderDecoderDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset for an ``EncoderDecoder`` model.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        Tokenizer to use to split the input sequences into words or other kinds of tokens.
    target_tokenizer : ``Tokenizer``, optional (default=``WordTokenizer(start_tokens=[START_SYMBOL],
                                                                        end_tokens=[END_SYMBOL])``)
        Tokenizer to use to split the output sequences (during training) into words or other kinds of tokens.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
                             (default=``{"tokens": SingleIdTokenizer("input_tokens")}``)
        Indexers used to define input (source side) token representations.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
                            (default=``{"tokens": SingleIdTokenizer("output_tokens")}``)
        Indexers used to define output (target side) token representations.
    """
    def __init__(self, source_tokenizer: Tokenizer = None, target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or WordTokenizer(start_tokens=[START_SYMBOL],
                                                                   end_tokens=[END_SYMBOL])
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer("source_tokens")}
        self._target_token_indexers = target_token_indexers or {"tokens": SingleIdTokenIndexer("target_tokens")}
        self._token_indexers = {"source": self._source_token_indexers, "target": self._target_token_indexers}

    @overrides
    def read(self, file_path):
        instances = []
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(tqdm.tqdm(data_file)):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                source_sequence, target_sequence = line_parts
                source_field = TextField(self._source_tokenizer.tokenize(source_sequence),
                                         self._source_token_indexers)
                target_field = TextField(self._target_tokenizer.tokenize(target_sequence),
                                         self._target_token_indexers)
                instances.append(Instance({'source_tokens': source_field, 'target_tokens': target_field}))
        if not instances:
            raise ConfigurationError("No instances read!")
        return Dataset(instances)

    @overrides
    def text_to_instance(self, input_string: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._source_tokenizer.tokenize(input_string)
        input_field = TextField(tokenized_string, self._source_token_indexers)
        return Instance({'source_tokens': input_field})

    @classmethod
    def from_params(cls, params: Params) -> 'EncoderDecoderDatasetReader':
        source_tokenizer = Tokenizer.from_params(params.pop('source_tokenizer', {}))
        target_tokenizer = Tokenizer.from_params(params.pop('target_tokenizer', {"start_tokens": [START_SYMBOL],
                                                                                 "end_tokens": [END_SYMBOL]}))
        source_token_indexers = TokenIndexer.dict_from_params(params.pop('source_token_indexers', {}))
        target_token_indexers = TokenIndexer.dict_from_params(params.pop('target_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return EncoderDecoderDatasetReader(source_tokenizer, target_tokenizer,
                                           source_token_indexers, target_token_indexers)
