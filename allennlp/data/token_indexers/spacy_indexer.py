from typing import Dict, List

from overrides import overrides
from spacy.tokens import Token as SpacyToken
import torch
import numpy

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList


@TokenIndexer.register("spacy")
class SpacyTokenIndexer(TokenIndexer):
    """
    This :class:`SpacyTokenIndexer` represents tokens as word vectors
    from a spacy model. You might want to do this for two main reasons;
    easier integration with a spacy pipeline and no out of vocabulary
    tokens.

    Registered as a `TokenIndexer` with name "spacy".

    # Parameters

    hidden_dim : `int`, optional (default=`96`)
        The dimension of the vectors that spacy generates for
        representing words.
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(self, hidden_dim: int = 96, token_min_padding_length: int = 0) -> None:
        self._hidden_dim = hidden_dim
        super().__init__(token_min_padding_length)

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # We are using spacy to generate embeddings directly for our model,
        # so we don't need to capture the vocab - it is defined by the spacy
        # model we are using instead.
        pass

    @overrides
    def tokens_to_indices(
        self, tokens: List[SpacyToken], vocabulary: Vocabulary
    ) -> Dict[str, List[numpy.ndarray]]:
        if not all(isinstance(x, SpacyToken) for x in tokens):
            raise ValueError(
                "The spacy indexer requires you to use a Tokenizer which produces SpacyTokens."
            )
        indices: List[numpy.ndarray] = [token.vector for token in tokens]
        return {"tokens": indices}

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        def padding_token():
            return numpy.zeros(self._hidden_dim, dtype=numpy.float32)

        tensor = torch.FloatTensor(
            pad_sequence_to_length(
                tokens["tokens"], padding_lengths["tokens"], default_value=padding_token
            )
        )
        return {"tokens": tensor}
