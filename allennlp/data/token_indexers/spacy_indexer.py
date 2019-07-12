from typing import Dict, List

from overrides import overrides
from spacy.tokens import Token as SpacyToken
import torch
import numpy

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer


@TokenIndexer.register("spacy")
class SpacyTokenIndexer(TokenIndexer[numpy.ndarray]):
    """
    This :class:`SpacyTokenIndexer` represents tokens as word vectors
    from a spacy model. You might want to do this for two main reasons;
    easier integration with a spacy pipeline and no out of vocabulary
    tokens.

    Parameters
    ----------
    hidden_dim : ``int``, optional (default=``96``)
        The dimension of the vectors that spacy generates for
        representing words.
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
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]): # pylint: disable=unused-argument
        # We are using spacy to generate embeddings directly for our model,
        # so we don't need to capture the vocab - it is defined by the spacy
        # model we are using instead.
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[SpacyToken],
                          vocabulary: Vocabulary, # pylint: disable=unused-argument
                          index_name: str) -> Dict[str, List[numpy.ndarray]]:

        if not all([isinstance(x, SpacyToken) for x in tokens]):
            raise ValueError("The spacy indexer requires you to use a Tokenizer which produces SpacyTokens.")
        indices: List[numpy.ndarray] = []
        for token in tokens:
            indices.append(token.vector)

        return {index_name: indices}

    def get_padding_token(self) -> numpy.ndarray:
        return numpy.zeros(self._hidden_dim, dtype=numpy.float32)

    @overrides
    def get_padding_lengths(self,
                            token: numpy.ndarray) -> Dict[str, numpy.ndarray]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[numpy.ndarray]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument

        val = {key: torch.FloatTensor(pad_sequence_to_length(
                val, desired_num_tokens[key], default_value=self.get_padding_token))
               for key, val in tokens.items()}
        return val
