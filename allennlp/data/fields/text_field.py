"""
A ``TextField`` represents a string of text, the kind that you might want to represent with
standard word vectors, or pass through an LSTM.
"""
from typing import Dict, List, Optional

from overrides import overrides
from spacy.tokens import Token as SpacyToken
import torch
from torch.autograd import Variable

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
            # Iterate over the keys and find the maximum token length.
            # It's fine to iterate over the keys of the first token since all tokens have the same keys.
            for key in token_lengths[0].keys():
                indexer_lengths[key] = max(x[key] if key in x else 0 for x in token_lengths)
            lengths.append(indexer_lengths)
        any_indexed_token_key = list(self._indexed_tokens.keys())[0]
        padding_lengths = {'num_tokens': len(self._indexed_tokens[any_indexed_token_key])}
        # Get all keys which have been used for padding for each indexer and take the max if there are duplicates.
        padding_keys = {key for d in lengths for key in d.keys()}
        for padding_key in padding_keys:
            padding_lengths[padding_key] = max(x[padding_key] if padding_key in x else 0 for x in lengths)
        return padding_lengths

    @overrides
    def sequence_length(self) -> int:
        return len(self.tokens)

    @overrides
    def as_tensor(self,
                  padding_lengths: Dict[str, int],
                  cuda_device: int = -1,
                  for_training: bool = True) -> Dict[str, torch.Tensor]:
        tensors = {}
        desired_num_tokens = padding_lengths['num_tokens']
        for indexer_name, indexer in self._token_indexers.items():
            padded_array = indexer.pad_token_sequence(self._indexed_tokens[indexer_name],
                                                      desired_num_tokens, padding_lengths)
            # We use the key of the indexer to recognise what the tensor corresponds to within the
            # field (i.e. the result of word indexing, or the result of character indexing, for
            # example).
            # TODO(mattg): we might someday have a TokenIndexer that needs to use something other
            # than a LongTensor here, and it's not clear how to signal that.  Maybe we'll need to
            # add a class method to TokenIndexer to tell us the type?  But we can worry about that
            # when there's a compelling use case for it.
            tensor = Variable(torch.LongTensor(padded_array), volatile=not for_training)
            tensors[indexer_name] = tensor if cuda_device == -1 else tensor.cuda(cuda_device)
        return tensors

    @overrides
    def empty_field(self):
        # pylint: disable=protected-access
        text_field = TextField([], self._token_indexers)
        # This needs to be a dict of empty lists for each token_indexer,
        # for padding reasons in ListField.
        text_field._indexed_tokens = {name: [] for name in self._token_indexers.keys()}
        return text_field

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=no-self-use
        # This is creating a dict of {token_indexer_key: batch_tensor} for each token indexer used
        # to index this field.
        return util.batch_tensor_dicts(tensor_list)
