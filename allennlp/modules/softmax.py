"""
The Softmax from calypso
"""

import numpy as np
import torch

class Softmax(torch.nn.Module):
    def __init__(self,
                 num_words: int,
                 embedding_dim: int,
                 tie_embeddings: bool = False,
                 token_encoder=None) -> None:
        super().__init__()

        self.tie_embeddings = tie_embeddings

        # Glorit init (std=(1.0 / sqrt(fan_in))
        if self.tie_embeddings:
            self.softmax_w = token_encoder
            # +1 for shape to include padding dimension
            self.softmax_b = torch.nn.Parameter(torch.zeros(num_words + 1))
        else:
            self.softmax_w = torch.nn.Parameter(
                    torch.randn(embedding_dim, num_words) / np.sqrt(embedding_dim)
            )
            self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        # embeddings is size (n, embedding_dim)
        # targets is (batch_size, ) with the correct class id
        # Does not do any count normalization / divide by batch size
        if self.tie_embeddings:
            softmax_w = self.softmax_w.weight.t()
        else:
            softmax_w = self.softmax_w

        probs = torch.nn.functional.log_softmax(
                torch.matmul(embeddings, softmax_w) + self.softmax_b,
                dim=-1
        )

        if self.tie_embeddings:
            # need to add back in padding dim!
            targets_ = targets + 1
        else:
            targets_ = targets

        return torch.nn.functional.nll_loss(probs, targets_.long(),
                                            reduction="sum")
