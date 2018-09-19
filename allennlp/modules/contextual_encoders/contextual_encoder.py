from typing import Dict

import torch

from allennlp.common.registrable import Registrable

class ContextualEncoder(Registrable, torch.nn.Module):
    """
    A ``ContextualEncoder`` takes token embeddings that are
        (batch_size, sequence_length, embedding_dim)
    and returns the
        (batch_size, sequence_length, contextual_encoding_dim)
    tensor of contextual encodings.
    """
    def __init__(self, num_layers: int, output_dim: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim

    def forward(self,
                token_embeddings: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        raise NotImplementedError

    def get_regularization_penalty(self) -> torch.Tensor:
        # pylint: disable=no-self-use
        return 0.0
