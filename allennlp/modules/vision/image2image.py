from torch import nn, FloatTensor

from allennlp.common import Registrable


class Image2ImageModule(nn.Module, Registrable):
    """
    An Image2ImageModule takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), and returns a tensor in the same format, after
    applying some transformation on the images.
    """

    def forward(self, images: FloatTensor):
        raise NotImplementedError()


@Image2ImageModule.register("null")
class NullImage2ImageModule(Image2ImageModule):
    """An `Image2ImageModule` that returns the original image unchanged."""

    def forward(self, images: FloatTensor):
        return images
