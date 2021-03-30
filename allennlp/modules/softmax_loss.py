import torch
import numpy as np


class SoftmaxLoss(torch.nn.Module):
    """
    Given some embeddings and some targets, applies a linear layer
    to create logits over possible words and then returns the
    negative log likelihood. Does not add a padding ID into the
    vocabulary, and input `targets` to `forward` should not include
    a padding ID.
    """

    def __init__(self, num_words: int, embedding_dim: int) -> None:
        super().__init__()

        # TODO(joelgrus): implement tie_embeddings (maybe)
        self.tie_embeddings = False

        self.softmax_w = torch.nn.Parameter(
            torch.randn(embedding_dim, num_words) / np.sqrt(embedding_dim)
        )
        self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        # Parameters

        embeddings : `torch.Tensor`
            A tensor of shape `(sequence_length, embedding_dim)`
        targets : `torch.Tensor`
            A tensor of shape `(batch_size, )`

        # Returns

        loss : `torch.FloatTensor`
            A scalar loss to be optimized.
        """
        # embeddings is size (n, embedding_dim)
        # targets is (batch_size, ) with the correct class id
        # Does not do any count normalization / divide by batch size
        probs = torch.nn.functional.log_softmax(
            torch.matmul(embeddings, self.softmax_w) + self.softmax_b, dim=-1
        )

        return torch.nn.functional.nll_loss(probs, targets.long(), reduction="sum")
