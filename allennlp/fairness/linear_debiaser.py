import torch


class LinearDebiaser(torch.nn.Module):
    """
    Linear debiaser. Debiases by removing component of embeddings
    in the bias direction.
    """

    def forward(
        self, embeddings: torch.Tensor, bias_direction: torch.Tensor, no_grad: bool = False
    ):
        """

        # Parameters

        embeddings : `torch.Tensor`
            A tensor of size (batch_size, ..., dim).
        bias_direction : `torch.Tensor`
            A tensor of size (dim, ).
        no_grad : `bool`, optional (default=`False`)
            Option to disable gradient calculation.

        # Returns

        debiased_embeddings : `torch.Tensor`
            A tensor of the same size as embeddings. debiased_embeddings do not contain a component
            in bias_direction.
        """
        with torch.set_grad_enabled(not no_grad):
            return (
                embeddings
                - torch.matmul(embeddings, bias_direction.reshape(-1, 1)) * bias_direction
            )
