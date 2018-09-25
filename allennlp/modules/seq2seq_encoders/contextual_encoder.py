from typing import Optional

import torch

from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

class ContextualEncoder(Seq2SeqEncoder):
    """
    The ``ContextualEncoder`` takes the output of a token embedder
    and applies a "context-adding" ``Seq2SeqEncoder`` to its output.

    Parameters
    ----------
    contextual_encoder: ``Seq2SeqEncoder``
        A ``Seq2SeqEncoder`` that adds "context" to the embedded tokens.
    num_layers: ``int``
        The total number of contextual layers + 1 (for the output layer).
    dropout: ``float``, optional (default = None)
        If specified, this will be applied immediately after the base embedder.
    embedding_layer_norm: ``MaskedLayerNorm``, optional (default = None)
        If provided, this will be applied to the embeddings after dropout
        but before the contextual encoder.
    return_all_layers: bool, optional (default = False)
        Should this module return all layers or only the last layer?
    """
    def __init__(self,
                 contextual_encoder: Seq2SeqEncoder,
                 num_layers: int,
                 dropout: float = None,
                 embedding_layer_norm: Optional[MaskedLayerNorm] = None,
                 return_all_layers: bool = False) -> None:
        super().__init__()
        self._contextual_encoder = contextual_encoder
        self._embedding_layer_norm = embedding_layer_norm
        self._num_layers = num_layers

        self.return_all_layers = return_all_layers

        if dropout is not None:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

    def forward(self, embeddings: torch.Tensor, mask: torch.LongTensor = None) -> torch.Tensor:
        # pylint: disable=arguments-differ
        if self._dropout is not None:
            embeddings = self._dropout(embeddings)

        if self._embedding_layer_norm is not None:
            embeddings = self._embedding_layer_norm(embeddings, mask)

        contextual_output = self._contextual_encoder(embeddings, mask)

        if self.return_all_layers:
            # Concatenate all layers with the token layer
            token_layer = torch.cat([embeddings, embeddings], dim=-1)
            contextual_layers = torch.cat(
                    [layer.unsqueeze(1) for layer in contextual_output], dim=-1
            )
            contextual_output = torch.cat(
                    [token_layer.unsqueeze(1), contextual_layers], dim=-1
            )

        return contextual_output

    def is_bidirectional(self) -> bool:
        return self._contextual_encoder.is_bidirectional()

    def get_input_dim(self) -> int:
        return self._contextual_encoder.get_input_dim()

    def get_output_dim(self) -> int:
        return self._contextual_encoder.get_output_dim()
