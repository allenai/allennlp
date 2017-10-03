from typing import Dict
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers import DatasetReader
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
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        Tokenizer to use to split the words in the input and output sequences.
    input_token_indexers : ``Dict[str, TokenIndexer]``, optional
                             (default=``{"tokens": SingleIdTokenizer("input_tokens")}``)
        Indexers used to define input (source side) token representations.
    output_token_indexers : ``Dict[str, TokenIndexer]``, optional
                            (default=``{"tokens": SingleIdTokenizer("output_tokens")}``)
        Indexers used to define output (target side) token representations.
    """
    def __init__(self, tokenizer: Tokenizer = None, input_token_indexers: Dict[str, TokenIndexer] = None,
                 output_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._input_token_indexers = input_token_indexers or {"tokens": SingleIdTokenIndexer("input_tokens")}
        self._output_token_indexers = output_token_indexers or {"tokens": SingleIdTokenIndexer("output_tokens")}
        self._token_indexers = {"source": self._input_token_indexers, "target": self._output_token_indexers}

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
                target_sequence = "%s %s %s" % (START_SYMBOL, target_sequence, END_SYMBOL)
                source_field = TextField(self._tokenizer.tokenize(source_sequence),
                                         self._input_token_indexers)
                target_field = TextField(self._tokenizer.tokenize(target_sequence),
                                         self._output_token_indexers)
                instances.append(Instance({'input_tokens': source_field, 'output_tokens': target_field}))
        if not instances:
            raise ConfigurationError("No instances read!")
        return Dataset(instances)

    @overrides
    def text_to_instance(self, input_string: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._tokenizer.tokenize(input_string)
        input_field = TextField(tokenized_string, self._input_token_indexers)
        return Instance({'input_tokens': input_field})

    @classmethod
    def from_params(cls, params: Params) -> 'EncoderDecoderDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        input_token_indexers = TokenIndexer.dict_from_params(params.pop('input_token_indexers', {}))
        output_token_indexers = TokenIndexer.dict_from_params(params.pop('output_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return EncoderDecoderDatasetReader(tokenizer, input_token_indexers, output_token_indexers)
