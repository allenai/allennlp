"""
A ``TextField`` represents a string of text, the kind that you might want to represent with
standard word vectors, or pass through an LSTM.
"""
from typing import Dict, List, Optional, Iterator
import textwrap

from overrides import overrides
from spacy.tokens import Token as SpacyToken
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

TokenList = List[TokenType]  # pylint: disable=invalid-name


class TextField(SequenceField[Dict[str, torch.Tensor]]):
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
        self._indexer_name_to_indexed_token: Optional[Dict[str, List[str]]] = None
        self._token_index_to_indexer_name: Optional[Dict[str, str]] = None

        if not all([isinstance(x, (Token, SpacyToken)) for x in tokens]):
            raise ConfigurationError("TextFields must be passed Tokens. "
                                     "Found: {} with types {}.".format(tokens, [type(x) for x in tokens]))

    # Sequence[Token] methods
    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self._token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)

    @overrides
    def index(self, vocab: Vocabulary):
        token_arrays: Dict[str, TokenList] = {}
        indexer_name_to_indexed_token: Dict[str, List[str]] = {}
        token_index_to_indexer_name: Dict[str, str] = {}
        for indexer_name, indexer in self._token_indexers.items():
            token_indices = indexer.tokens_to_indices(self.tokens, vocab, indexer_name)
            token_arrays.update(token_indices)
            indexer_name_to_indexed_token[indexer_name] = list(token_indices.keys())
            for token_index in token_indices:
                token_index_to_indexer_name[token_index] = indexer_name
        self._indexed_tokens = token_arrays
        self._indexer_name_to_indexed_token = indexer_name_to_indexed_token
        self._token_index_to_indexer_name = token_index_to_indexer_name

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        The ``TextField`` has a list of ``Tokens``, and each ``Token`` gets converted into arrays by
        (potentially) several ``TokenIndexers``.  This method gets the max length (over tokens)
        associated with each of these arrays.
        """
        # Our basic outline: we will iterate over `TokenIndexers`, and aggregate lengths over tokens
        # for each indexer separately.  Then we will combine the results for each indexer into a single
        # dictionary, resolving any (unlikely) key conflicts by taking a max.
        lengths = []
        if self._indexed_tokens is None:
            raise ConfigurationError("You must call .index(vocabulary) on a "
                                     "field before determining padding lengths.")

        # Each indexer can return a different sequence length, and for indexers that return
        # multiple arrays each can have a different length.  We'll keep track of them here.
        for indexer_name, indexer in self._token_indexers.items():
            indexer_lengths = {}

            for indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]:
                # This is a list of dicts, one for each token in the field.
                token_lengths = [indexer.get_padding_lengths(token)
                                 for token in self._indexed_tokens[indexed_tokens_key]]
                if not token_lengths:
                    # This is a padding edge case and occurs when we want to pad a ListField of
                    # TextFields. In order to pad the list field, we need to be able to have an
                    # _empty_ TextField, but if this is the case, token_lengths will be an empty
                    # list, so we add the padding for a token of length 0 to the list instead.
                    token_lengths = [indexer.get_padding_lengths([])]
                # Iterate over the keys and find the maximum token length.
                # It's fine to iterate over the keys of the first token since all tokens have the same keys.
                for key in token_lengths[0]:
                    indexer_lengths[key] = max(x[key] if key in x else 0 for x in token_lengths)
            lengths.append(indexer_lengths)

        padding_lengths = {}
        num_tokens = set()
        for token_index, token_list in self._indexed_tokens.items():
            indexer_name = self._token_index_to_indexer_name[token_index]
            indexer = self._token_indexers[indexer_name]
            padding_lengths[f"{token_index}_length"] = max(len(token_list),
                                                           indexer.get_token_min_padding_length())
            num_tokens.add(len(token_list))

        # We don't actually use this for padding anywhere, but we used to.  We add this key back in
        # so that older configs still work if they sorted by this key in a BucketIterator.  Taking
        # the max of all of these should be fine for that purpose.
        padding_lengths['num_tokens'] = max(num_tokens)

        # Get all keys which have been used for padding for each indexer and take the max if there are duplicates.
        padding_keys = {key for d in lengths for key in d.keys()}
        for padding_key in padding_keys:
            padding_lengths[padding_key] = max(x.get(padding_key, 0) for x in lengths)
        return padding_lengths

    @overrides
    def sequence_length(self) -> int:
        return len(self.tokens)

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        tensors = {}
        for indexer_name, indexer in self._token_indexers.items():
            desired_num_tokens = {indexed_tokens_key: padding_lengths[f"{indexed_tokens_key}_length"]
                                  for indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]}
            indices_to_pad = {indexed_tokens_key: self._indexed_tokens[indexed_tokens_key]
                              for indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]}

            indexer_tensors = indexer.as_padded_tensor(indices_to_pad,
                                                       desired_num_tokens,
                                                       padding_lengths)
            # We use the key of the indexer to recognise what the tensor corresponds to within the
            # field (i.e. the result of word indexing, or the result of character indexing, for
            # example).
            tensors.update(indexer_tensors)
        return tensors

    @overrides
    def empty_field(self):
        # pylint: disable=protected-access
        text_field = TextField([], self._token_indexers)
        text_field._indexed_tokens = {}
        text_field._indexer_name_to_indexed_token = {}
        for indexer_name, indexer in self._token_indexers.items():
            array_keys = indexer.get_keys(indexer_name)
            for key in array_keys:
                text_field._indexed_tokens[key] = []
            text_field._indexer_name_to_indexed_token[indexer_name] = array_keys
        return text_field

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=no-self-use
        # This is creating a dict of {token_indexer_key: batch_tensor} for each token indexer used
        # to index this field.
        return util.batch_tensor_dicts(tensor_list)

    def __str__(self) -> str:
        indexers = {name: indexer.__class__.__name__ for name, indexer in self._token_indexers.items()}

        # Double tab to indent under the header.
        formatted_text = "".join(["\t\t" + text + "\n"
                                  for text in textwrap.wrap(repr(self.tokens), 100)])
        return f"TextField of length {self.sequence_length()} with " \
               f"text: \n {formatted_text} \t\tand TokenIndexers : {indexers}"
