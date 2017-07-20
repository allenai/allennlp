from typing import Dict

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.experiments import Registry
from allennlp.modules import TokenEmbedder, TokenVectorizer


@Registry.register_token_embedder("basic")
class BasicTokenEmbedder(TokenEmbedder):
    """
    This is a ``TokenEmbedder`` that wraps a collection of :class:`TokenVectorizer` objects.  Each
    ``TokenVectorizer`` embeds or encodes the representation output from one
    :class:`~allennlp.data.TokenIndexer`.  As the data produced by a
    :class:`~allennlp.data.fields.TextField` is a dictionary mapping names to these
    representations, we take ``TokenVectorizers`` with corresponding names.  Each
    ``TokenVectorizer`` embeds its input, and the result is concatenated in an arbitrary order.
    """
    def __init__(self, token_vectorizers: Dict[str, TokenVectorizer]) -> None:
        super(BasicTokenEmbedder, self).__init__()
        self._token_vectorizers = token_vectorizers

    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        for vectorizer in self._token_vectorizers.values():
            output_dim += vectorizer.get_output_dim()
        return output_dim

    def forward(self, text_field_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self._token_vectorizers.keys() == text_field_input.keys(), "mismatched token keys"
        embedded_representations = []
        for key, tensor in text_field_input.items():
            vectorizer = self._token_vectorizers[key]
            token_vectors = vectorizer(tensor)
            embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'TokenEmbedder':
        token_vectorizers = {}
        for key, vectorizer_params in params.items():
            token_vectorizers[key] = TokenVectorizer.from_params(vocab, vectorizer_params)
        return cls(token_vectorizers)
