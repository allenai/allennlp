from typing import Dict

import torch

from allennlp.modules.contextual_encoders.contextual_encoder import ContextualEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

@ContextualEncoder.register('contextual-seq2seq')
class ContextualSeq2SeqEncoder(ContextualEncoder):
    """
    A ``ContextualEncoder`` that just wraps an AllenNLP ``Seq2SeqEncoder``.

    Parameters
    ----------
    encoder : ``Seq2SeqEncoder``
        An AllenNLP ``Seq2SeqEncoder``.
    num_layers : int, optional (default: 1)
        How many layers to the Seq2SeqEncoder
    dropout : float, optional (default: None)
        If specified, this dropout is applied before the encoder.
    """
    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 num_layers: int = 1,
                 dropout: float = None) -> None:
        super().__init__(num_layers=num_layers,
                         output_dim=encoder.get_output_dim())
        self._encoder = encoder
        if dropout is not None:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

    def forward(self,
                token_embeddings: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._dropout:
            return self._encoder(self._dropout(token_embeddings, mask))
        else:
            return self._encoder(token_embeddings, mask)
