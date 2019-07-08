from typing import Dict, List
import itertools

from overrides import overrides
from spacy.tokens import Token as SpacyToken
import torch
import numpy

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer


@TokenIndexer.register("spacy")
class SpacyTokenIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` represents tokens as single integers.

    Parameters
    ----------
    namespace : ``str``, optional (default=``tokens``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    lowercase_tokens : ``bool``, optional (default=``False``)
        If ``True``, we will call ``token.lower()`` before getting an index for the token from the
        vocabulary.
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to ``tokens_to_indices``.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to ``tokens_to_indices``.
    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 hidden_dim: int = 96,
                 token_min_padding_length: int = 0) -> None:
        self._hidden_dim = hidden_dim
        super().__init__(token_min_padding_length)

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:

        if not all([isinstance(x, SpacyToken) for x in tokens]):
            raise ValueError("The spacy indexer requires you to use a Tokenizer which produces SpacyTokens.")
        indices: List[int] = []
        for token in tokens:
            indices.append(token.vector)

        return {index_name: indices}

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def array_type(self):
        return torch.FloatTensor

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument

        def zeros(): return numpy.zeros(self._hidden_dim, dtype=numpy.float32)
        val = {key: pad_sequence_to_length(val, desired_num_tokens[key], default_value=zeros)
                for key, val in tokens.items()}
        return val
