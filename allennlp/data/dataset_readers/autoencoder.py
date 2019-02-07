import logging
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Token
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import TextField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("autoencoder")
class AutoencoderDatasetReader(Seq2SeqDatasetReader):
    """
    ``AutoencoderDatasetReader`` class inherits Seq2SeqDatasetReader as the only difference is when dealing with autoencoding tasks i.e., the target equals the source.
    """
    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for _, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                yield self.text_to_instance(line)

    @overrides
    def text_to_instance(self, input_string: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized_string = self._source_tokenizer.tokenize(input_string)
        tokenized_source = tokenized_string.copy()
        tokenized_target = tokenized_string.copy()
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        target_field = TextField(tokenized_target, self._target_token_indexers)
        return Instance({"source_tokens": source_field, "target_tokens": target_field})
