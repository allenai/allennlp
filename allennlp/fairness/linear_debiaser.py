import torch

# TODO: bias_direction could have rank > 1
class LinearDebiaser(torch.nn.Module):
    """
    Linear debiaser. Debiases embeddings by removing component
    in the bias direction.

    Implementation and terminology based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
    V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021).
    [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
    Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
    ArXiv, abs/2104.02797.
    """

    def __init__(self, requires_grad: bool = False):
        """

        # Parameters

        requires_grad : `bool`, optional (default=`False`)
            Option to enable gradient calculation.
        """
        self.requires_grad = requires_grad

    def forward(self, embeddings: torch.Tensor, bias_direction: torch.Tensor):
        """

        # Parameters

        embeddings : `torch.Tensor`
            A tensor of size (batch_size, ..., dim).
        bias_direction : `torch.Tensor`
            A tensor of size (dim, ).

        # Returns

        debiased_embeddings : `torch.Tensor`
            A tensor of the same size as embeddings. debiased_embeddings do not contain a component
            in bias_direction.
        """
        with torch.set_grad_enabled(self.requires_grad):
            return (
                embeddings
                - torch.matmul(embeddings, bias_direction.reshape(-1, 1)) * bias_direction
            )
