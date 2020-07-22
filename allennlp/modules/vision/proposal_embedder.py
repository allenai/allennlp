import torch
from torch import nn, FloatTensor

from allennlp.common.registrable import Registrable


class ProposalEmbedder(nn.Module, Registrable):
    """
    An `ProposalEmbedder` takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), together with a batch of proposals as a tensor with the
    dimensions (Batch, Boxes, (x1,y1,x2,y2)), and returns an embedding of each proposal in a tensor
    with the dimensions (Batch, Boxes, Features).
    """

    def forward(self, images: FloatTensor, proposals: FloatTensor) -> FloatTensor:
        raise NotImplementedError()

    def get_output_dim(self) -> int:
        """
        Returns the final output dimension that this `ProposalEmbedder` uses to represent each
        proposal. This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError


@ProposalEmbedder.register("null")
class NullProposalEmbedder(ProposalEmbedder):
    """A `ProposalEmbedder` that returns constants regardless of the input."""

    def __init__(self, output_dim: int, constant: float = 0.0):
        super().__init__()
        self.output_dim = output_dim
        self.constant = constant

    def forward(self, images: FloatTensor, proposals: FloatTensor) -> FloatTensor:
        assert images.size(0) == proposals.size(0)
        return torch.full(
            (images.size(0), proposals.size(1), self.get_output_dim()),
            fill_value=self.constant,
            dtype=torch.float32,
            device=images.device,
        )

    def get_output_dim(self) -> int:
        return self.output_dim
