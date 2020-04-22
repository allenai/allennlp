import torch

from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("character_encoding")
class TokenCharactersEncoder(TokenEmbedder):
    """
    A `TokenCharactersEncoder` takes the output of a
    [`TokenCharactersIndexer`](../../data/token_indexers/token_characters_indexer.md), which is a tensor of shape
    (batch_size, num_tokens, num_characters), embeds the characters, runs a token-level encoder, and
    returns the result, which is a tensor of shape (batch_size, num_tokens, encoding_dim).  We also
    optionally apply dropout after the token-level encoder.

    We take the embedding and encoding modules as input, so this class is itself quite simple.

    Registered as a `TokenEmbedder` with name "character_encoding".
    """

    def __init__(self, embedding: Embedding, encoder: Seq2VecEncoder, dropout: float = 0.0) -> None:
        super().__init__()
        self._embedding = TimeDistributed(embedding)
        self._encoder = TimeDistributed(encoder)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    def get_output_dim(self) -> int:
        return self._encoder._module.get_output_dim()

    def forward(self, token_characters: torch.Tensor) -> torch.Tensor:
        mask = (token_characters != 0).long()
        return self._dropout(self._encoder(self._embedding(token_characters), mask))
