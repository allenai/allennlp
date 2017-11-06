"""
A ``TextField`` represents a string of text, the kind that you might want to represent with
standard word vectors, or pass through an LSTM.
"""
from typing import Dict, List, Optional

from overrides import overrides
from spacy.tokens import Token as SpacyToken
import numpy

from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

TokenList = List[TokenType]  # pylint: disable=invalid-name


class TextField(SequenceField[Dict[str, numpy.ndarray]]):
    """
    This ``Field`` represents a list of string tokens.  Before constructing this object, you need
    to tokenize raw strings using a :class:`~allennlp.data.tokenizers.tokenizer.Tokenizer`.

    Because string tokens can be represented as indexed arrays in a number of ways, we also take a
    dictionary of :class:`~allennlp.data.token_indexers.token_indexer.TokenIndexer`
    objects that will be used to convert the tokens into indices.
    Each ``TokenIndexer`` could represent each token as a single ID, or a list of character IDs, or
    something else.

    This field will get converted into a dictionary of arrays, one for each ``TokenIndexer``.  A
    ``SingleIdTokenIndexer`` produces an array of shape (num_tokens,), while a
    ``TokenCharactersIndexer`` produces an array of shape (num_tokens, num_characters).
    """
    def __init__(self, tokens: List[Token], token_indexers: Dict[str, TokenIndexer]) -> None:
        self.tokens = tokens
        self._token_indexers = token_indexers
        self._indexed_tokens: Optional[Dict[str, TokenList]] = None

        if not all([isinstance(x, (Token, SpacyToken)) for x in tokens]):
            raise ConfigurationError("TextFields must be passed Tokens. "
                                     "Found: {} with types {}.".format(tokens, [type(x) for x in tokens]))

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self._token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)

    @overrides
    def index(self, vocab: Vocabulary):
        token_arrays = {}
        for indexer_name, indexer in self._token_indexers.items():
            arrays = [indexer.token_to_indices(token, vocab) for token in self.tokens]
            token_arrays[indexer_name] = arrays
        self._indexed_tokens = token_arrays

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        lengths = []
        if self._indexed_tokens is None:
            raise ConfigurationError("You must call .index(vocabulary) on a "
                                     "field before determining padding lengths.")
        for indexer_name, indexer in self._token_indexers.items():
            indexer_lengths = {}

            # This is a list of dicts, one for each token in the field.
            token_lengths = [indexer.get_padding_lengths(token) for token in self._indexed_tokens[indexer_name]]
            if not token_lengths:
                # This is a padding edge case and occurs when we want to pad a ListField of
                # TextFields. In order to pad the list field, we need to be able to have an
                # _empty_ TextField, but if this is the case, token_lengths will be an empty
                # list, so we add the default empty padding dictionary to the list instead.
                token_lengths = [{}]
            # Iterate over the keys in the first element of the list.
            # This is fine as for a given indexer, all tokens will return the same keys,
            # so we can just use the first one.
            for key in token_lengths[0].keys():
                indexer_lengths[key] = max(x[key] if key in x else 0 for x in token_lengths)
            lengths.append(indexer_lengths)
        any_indexed_token_key = list(self._indexed_tokens.keys())[0]
        padding_lengths = {'num_tokens': len(self._indexed_tokens[any_indexed_token_key])}
        # Get all the keys which have been used for padding.
        padding_keys = {key for d in lengths for key in d.keys()}
        for padding_key in padding_keys:
            padding_lengths[padding_key] = max(x[padding_key] if padding_key in x else 0 for x in lengths)
        return padding_lengths

    @overrides
    def sequence_length(self) -> int:
        return len(self.tokens)

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> Dict[str, numpy.ndarray]:
        arrays = {}
        desired_num_tokens = padding_lengths['num_tokens']
        for indexer_name, indexer in self._token_indexers.items():
            padded_array = indexer.pad_token_sequence(self._indexed_tokens[indexer_name],
                                                      desired_num_tokens, padding_lengths)
            # Use the key of the indexer to recognise what the array corresponds to within the field
            # (i.e. the result of word indexing, or the result of character indexing, for example).
            arrays[indexer_name] = numpy.array(padded_array)
        return arrays

    @overrides
    def empty_field(self):
        # pylint: disable=protected-access
        text_field = TextField([], self._token_indexers)
        # This needs to be a dict of empty lists for each token_indexer,
        # for padding reasons in ListField.
        text_field._indexed_tokens = {name: [] for name in self._token_indexers.keys()}
        return text_field
