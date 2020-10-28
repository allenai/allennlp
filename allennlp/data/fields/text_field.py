"""
A `TextField` represents a string of text, the kind that you might want to represent with
standard word vectors, or pass through an LSTM.
"""
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Iterator
import textwrap

from overrides import overrides
from spacy.tokens import Token as SpacyToken
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

# There are two levels of dictionaries here: the top level is for the *key*, which aligns
# TokenIndexers with their corresponding TokenEmbedders.  The bottom level is for the *objects*
# produced by a given TokenIndexer, which will be input to a particular TokenEmbedder's forward()
# method.  We label these as tensors, because that's what they typically are, though they could in
# reality have arbitrary type.
TextFieldTensors = Dict[str, Dict[str, torch.Tensor]]


class TextField(SequenceField[TextFieldTensors]):
    """
    This `Field` represents a list of string tokens.  Before constructing this object, you need
    to tokenize raw strings using a :class:`~allennlp.data.tokenizers.tokenizer.Tokenizer`.

    Because string tokens can be represented as indexed arrays in a number of ways, we also take a
    dictionary of :class:`~allennlp.data.token_indexers.token_indexer.TokenIndexer`
    objects that will be used to convert the tokens into indices.
    Each `TokenIndexer` could represent each token as a single ID, or a list of character IDs, or
    something else.

    This field will get converted into a dictionary of arrays, one for each `TokenIndexer`.  A
    `SingleIdTokenIndexer` produces an array of shape (num_tokens,), while a
    `TokenCharactersIndexer` produces an array of shape (num_tokens, num_characters).
    """

    __slots__ = ["tokens", "_token_indexers", "_indexed_tokens"]

    def __init__(self, tokens: List[Token], token_indexers: Dict[str, TokenIndexer]) -> None:
        self.tokens = tokens
        self._token_indexers = token_indexers
        self._indexed_tokens: Optional[Dict[str, IndexedTokenList]] = None

        if not all(isinstance(x, (Token, SpacyToken)) for x in tokens):
            raise ConfigurationError(
                "TextFields must be passed Tokens. "
                "Found: {} with types {}.".format(tokens, [type(x) for x in tokens])
            )

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self._token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)

    @overrides
    def index(self, vocab: Vocabulary):
        self._indexed_tokens = {}
        for indexer_name, indexer in self._token_indexers.items():
            self._indexed_tokens[indexer_name] = indexer.tokens_to_indices(self.tokens, vocab)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        The `TextField` has a list of `Tokens`, and each `Token` gets converted into arrays by
        (potentially) several `TokenIndexers`.  This method gets the max length (over tokens)
        associated with each of these arrays.
        """
        if self._indexed_tokens is None:
            raise ConfigurationError(
                "You must call .index(vocabulary) on a field before determining padding lengths."
            )

        padding_lengths = {}
        for indexer_name, indexer in self._token_indexers.items():
            indexer_lengths = indexer.get_padding_lengths(self._indexed_tokens[indexer_name])
            for key, length in indexer_lengths.items():
                padding_lengths[f"{indexer_name}___{key}"] = length
        return padding_lengths

    @overrides
    def sequence_length(self) -> int:
        return len(self.tokens)

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> TextFieldTensors:
        if self._indexed_tokens is None:
            raise ConfigurationError(
                "You must call .index(vocabulary) on a field before calling .as_tensor()"
            )

        tensors = {}

        indexer_lengths: Dict[str, Dict[str, int]] = defaultdict(dict)
        for key, value in padding_lengths.items():
            # We want this to crash if the split fails. Should never happen, so I'm not
            # putting in a check, but if you fail on this line, open a github issue.
            indexer_name, padding_key = key.split("___")
            indexer_lengths[indexer_name][padding_key] = value

        for indexer_name, indexer in self._token_indexers.items():
            tensors[indexer_name] = indexer.as_padded_tensor_dict(
                self._indexed_tokens[indexer_name], indexer_lengths[indexer_name]
            )
        return tensors

    @overrides
    def empty_field(self):
        text_field = TextField([], self._token_indexers)
        text_field._indexed_tokens = {}
        for indexer_name, indexer in self._token_indexers.items():
            text_field._indexed_tokens[indexer_name] = indexer.get_empty_token_list()
        return text_field

    @overrides
    def batch_tensors(self, tensor_list: List[TextFieldTensors]) -> TextFieldTensors:
        # This is creating a dict of {token_indexer_name: {token_indexer_outputs: batched_tensor}}
        # for each token indexer used to index this field.
        indexer_lists: Dict[str, List[Dict[str, torch.Tensor]]] = defaultdict(list)
        for tensor_dict in tensor_list:
            for indexer_name, indexer_output in tensor_dict.items():
                indexer_lists[indexer_name].append(indexer_output)
        batched_tensors = {
            # NOTE(mattg): if an indexer has its own nested structure, rather than one tensor per
            # argument, then this will break.  If that ever happens, we should move this to an
            # `indexer.batch_tensors` method, with this logic as the default implementation in the
            # base class.
            indexer_name: util.batch_tensor_dicts(indexer_outputs)
            for indexer_name, indexer_outputs in indexer_lists.items()
        }
        return batched_tensors

    def __str__(self) -> str:
        indexers = {
            name: indexer.__class__.__name__ for name, indexer in self._token_indexers.items()
        }

        # Double tab to indent under the header.
        formatted_text = "".join(
            "\t\t" + text + "\n" for text in textwrap.wrap(repr(self.tokens), 100)
        )
        return (
            f"TextField of length {self.sequence_length()} with "
            f"text: \n {formatted_text} \t\tand TokenIndexers : {indexers}"
        )

    # Sequence[Token] methods
    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    @overrides
    def duplicate(self):
        """
        Overrides the behavior of `duplicate` so that `self._token_indexers` won't
        actually be deep-copied.

        Not only would it be extremely inefficient to deep-copy the token indexers,
        but it also fails in many cases since some tokenizers (like those used in
        the 'transformers' lib) cannot actually be deep-copied.
        """
        new = TextField(deepcopy(self.tokens), {k: v for k, v in self._token_indexers.items()})
        new._indexed_tokens = deepcopy(self._indexed_tokens)
        return new
