"""
A ``TextField`` represents a string of text, the kind that you might want to represent with
standard word vectors, or pass through an LSTM.
"""
from typing import Dict, List, Optional  # pylint: disable=unused-import

from overrides import overrides
import numpy

from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers.token_indexer import TokenType  # pylint: disable=unused-import
from allennlp.data.token_indexers import TokenIndexer
from allennlp.common.checks import ConfigurationError


class TextField(SequenceField):
    """
    This ``Field`` represents a list of string tokens.  Before constructing this object, you need
    to tokenize raw strings using a :class:`..tokenizers.Tokenizer`.

    Because string tokens can be represented as indexed arrays in a number of ways, we also take a
    list of :class:`TokenIndexer` objects that will be used to convert the tokens into indices.
    Each ``TokenIndexer`` could represent each token as a single ID, or a list of character IDs, or
    something else.

    This field will get converted into a dictionary of arrays, one for each ``TokenIndexer``.  A
    ``SingleIdTokenIndexer`` produces an array of shape (num_tokens,), while a
    ``TokenCharactersIndexer`` produces an array of shape (num_tokens, num_characters).
    """
    def __init__(self, tokens: List[str], token_indexers: List[TokenIndexer]) -> None:
        self._tokens = tokens
        self._token_indexers = token_indexers
        self._indexed_tokens = None  # type: Optional[List[List[TokenType]]]

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self._token_indexers:
            for token in self._tokens:
                indexer.count_vocab_items(token, counter)

    @overrides
    def index(self, vocab: Vocabulary):
        token_arrays = []
        for indexer in self._token_indexers:
            arrays = [indexer.token_to_indices(token, vocab) for token in self._tokens]
            token_arrays.append(arrays)
        self._indexed_tokens = token_arrays

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        lengths = []
        if self._indexed_tokens is None:
            raise ConfigurationError("You must call .index(vocabulary) on a "
                                     "field before determining padding lengths.")
        for indexer, array in zip(self._token_indexers, self._indexed_tokens):
            indexer_lengths = {}

            # This is a list of dicts, one for each token in the field.
            token_lengths = [indexer.get_padding_lengths(token) for token in array]
            # TODO(Mark): This breaks if the token list is empty, but we need to be able to have empty fields.
            # Just raise here?
            # Iterate over the keys in the first element of the list.
            # This is fine as for a given indexer, all tokens will return the same keys,
            # so we can just use the first one.
            for key in token_lengths[0].keys():
                indexer_lengths[key] = max(x[key] if key in x else 0 for x in token_lengths)
            lengths.append(indexer_lengths)
        padding_lengths = {'num_tokens': len(self._indexed_tokens[0])}
        # Get all the keys which have been used for padding.
        padding_keys = {key for d in lengths for key in d.keys()}
        for padding_key in padding_keys:
            padding_lengths[padding_key] = max(x[padding_key] if padding_key in x else 0 for x in lengths)
        return padding_lengths

    @overrides
    def sequence_length(self) -> int:
        return len(self._tokens)

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> Dict[str, numpy.array]:
        arrays = {}
        desired_num_tokens = padding_lengths['num_tokens']
        for indexer, array in zip(self._token_indexers, self._indexed_tokens):
            padded_array = indexer.pad_token_sequence(array, desired_num_tokens, padding_lengths)
            # Use the namespace of the indexer as a key to recognise what
            # the array corresponds to in a model.
            arrays[indexer.namespace] = padded_array
        return arrays

    @overrides
    def empty_field(self):
        # pylint: disable=protected-access
        text_field = TextField([], self._token_indexers)
        # This needs to be a list of empty lists for each token_indexer,
        # for padding reasons in ListField.
        text_field._indexed_tokens = [[] for _ in range(len(self._token_indexers))]
        return text_field

    def tokens(self):
        return self._tokens
