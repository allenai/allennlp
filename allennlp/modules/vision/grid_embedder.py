from torch import nn, FloatTensor

from allennlp.common.registrable import Registrable


class GridEmbedder(nn.Module, Registrable):
    """
    A `GridEmbedder` takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), and returns a tensor in the format
    (Batch, Features, new_height, new_width).
    For every image, it embeds a patch of the image, and returns the embedding
    of the patch. The size of the image might change during this operation.
    """

    def forward(self, images: FloatTensor) -> FloatTensor:
        raise NotImplementedError()

    def get_output_dim(self) -> int:
        """
        Returns the output dimension that this `GridEmbedder` uses to represent each
        patch. This is `not` the shape of the returned tensor, but the second dimension
        of that shape.
        """
        raise NotImplementedError


@GridEmbedder.register("null")
class NullGridEmbedder(GridEmbedder):
    """A `GridEmbedder` that returns the input image as given."""

    def forward(self, images: FloatTensor) -> FloatTensor:
        return images

    def get_output_dim(self) -> int:
        return 3
