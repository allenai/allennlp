from typing import Dict, List
import logging

from overrides import overrides
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import ListField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("language_modeling")
class LanguageModelingReader(DatasetReader):
    """
    Reads a text file and converts it into a ``Dataset`` suitable for training
    a language model.

    Parameters
    ----------
    batch_size : ``int``, optional (default=``20``)
        Batch size to use in language modeling.
    truncated_bptt_size : ``int``, optional (default=``35``)
        The sequence length to use for truncated backpropagation through time.
    fuzz_truncated_bptt_size : ``bool``, optional (default=``True``)
        If True, randomly perturb the truncated_bptt_size between batches.
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for the text.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to each line read by the dataset reader.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to each line read by the dataset reader.
    """
    def __init__(self,
                 batch_size: int = 20,
                 truncated_bptt_size: int = 35,
                 fuzz_truncated_bptt_size: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = ["</S>"]) -> None:
        super().__init__(lazy=False)
        self._batch_size = batch_size
        if truncated_bptt_size < 2:
            raise ConfigurationError("truncated_bptt_size cannot be less than 2.")
        self._truncated_bptt_size = truncated_bptt_size
        self._fuzz_truncated_bptt_size = fuzz_truncated_bptt_size
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]
        # Cache the batched tokens we read from data files.
        self._all_batched_file_tokens = {}

    @overrides
    def _read(self, file_path: str):
        if file_path not in self._all_batched_file_tokens:
            logger.info('Loading data from %s', file_path)
            # if `file_path` is a URL, redirect to the cache
            file_path = cached_path(file_path)

            # Read the contents of the file into one long list of tokens,
            # adding start and/or end tokens as necessary.
            file_tokens = []
            file_tokens.extend(self._start_tokens)
            with open(file_path, "r") as text_file:
                for line in text_file:
                    tokenized_line = self._tokenizer.tokenize(line)
                    file_tokens.extend(tokenized_line)
                    file_tokens.extend(self._end_tokens)

            # Divide file_tokens into batch_size lists
            # Work out how we can evenly split the dataset into batch_size parts
            total_num_tokens_per_batch = len(file_tokens) // self._batch_size
            if total_num_tokens_per_batch == 0:
                # TODO (nfliu): figure out if this is the desired behavior
                raise ValueError(f"There are {len(file_tokens)} tokens in the file, "
                                 f"but batch size is {self._batch_size}. "
                                 "batch size must be less than or equal to number of "
                                 "tokens in the file.")

            # Trim off the remainder from file_tokens, so we can evenly divide it
            # into batch_size lists.
            file_tokens_for_even_split = file_tokens[:total_num_tokens_per_batch *
                                                     self._batch_size]
            # Evenly divide the data into batch_size lists.
            batched_file_tokens = [
                    file_tokens_for_even_split[i:i + total_num_tokens_per_batch] for i in
                    range(0, len(file_tokens_for_even_split), total_num_tokens_per_batch)]
            # Cache the tokens of the dataset we've just read.
            self._all_batched_file_tokens[file_path] = batched_file_tokens
        else:
            batched_file_tokens = self._all_batched_file_tokens[file_path]

        # Iterate over the batched_file_tokens, yielding batches
        batch_start_index = 0
        # The max value of batch_start_index is len(batched_file_tokens[0]) - 2,
        # leaving room for the target even when the final batch is size 1.
        while batch_start_index < len(batched_file_tokens[0]) - 1:
            if self._fuzz_truncated_bptt_size:
                # This randomization is taken from the code for training the AWD-LSTM.
                # (matrices of size (batch_size, truncated_bptt_size))
                fuzzy_truncated_bptt_size = (
                        self._truncated_bptt_size if numpy.random.random() < 0.95 else
                        self._truncated_bptt_size / 2.)
                # Prevent excessively small or negative sequence length
                sequence_length = max(5,
                                      int(numpy.random.normal(fuzzy_truncated_bptt_size, 5)))
                # There's a very small chance that it could select a very long sequence
                # length, resulting in OOM. So we cap it at no more than
                # self._truncated_bptt_size + 10
                sequence_length = min(sequence_length, self._truncated_bptt_size + 10)
            else:
                sequence_length = self._truncated_bptt_size

            # We need to constrain the sequence_length to ensure that
            # the targets don't reach beyond the length of our dataset
            sequence_length = min(sequence_length,
                                  len(batched_file_tokens[0]) - batch_start_index - 1)
            batch_inputs = [single_batch[batch_start_index:batch_start_index + sequence_length]
                            for single_batch in batched_file_tokens]
            batch_targets = [single_batch[batch_start_index + 1:batch_start_index + 1 + sequence_length]
                             for single_batch in batched_file_tokens]

            # Take the examples between batch_start_index and sequence_length
            yield Instance({
                    "inputs": ListField([TextField(single_batch, self._token_indexers) for
                                         single_batch in batch_inputs]),
                    "forward_targets": ListField([TextField(single_batch, self._token_indexers) for
                                                  single_batch in batch_targets])
            })
            batch_start_index += sequence_length
