import torch

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.experiments import Registry
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules import Seq2VecEncoder, TimeDistributed, TokenEmbedder


@Registry.register_token_embedder("character_encoding")
class TokenCharactersEncoder(TokenEmbedder):
    """
    A ``TokenCharactersEncoder`` takes the output of a
    :class:`~allennlp.data.token_indexers.TokenCharactersIndexer`, which is a tensor of shape
    (batch_size, num_tokens, num_characters), embeds the characters, runs a token-level encoder, and
    returns the result, which is a tensor of shape (batch_size, num_tokens, encoding_dim).

    We take the embedding and encoding modules as input, so this class is itself quite simple.
    """
    def __init__(self, embedding: Embedding, encoder: Seq2VecEncoder) -> None:
        super(TokenCharactersEncoder, self).__init__()
        self._embedding = TimeDistributed(embedding)
        self._encoder = TimeDistributed(encoder)

    def get_output_dim(self) -> int:
        return self._encoder._module.get_output_dim()  # pylint: disable=protected-access

    def forward(self, token_characters: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        return self._encoder(self._embedding(token_characters))

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        embedding_params = params.pop("embedding")  # type: Params
        # Embedding.from_params() uses "tokens" as the default namespace, but we need to change
        # that to be "token_characters" by default.
        embedding_params.setdefault("vocab_namespace", "token_characters")
        embedding = Embedding.from_params(vocab, embedding_params)
        encoder_params = params.pop("encoder")  # type: Params
        encoder = Seq2VecEncoder.from_params(encoder_params)
        return cls(embedding, encoder)
