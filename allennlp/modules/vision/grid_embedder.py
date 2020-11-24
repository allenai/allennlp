import torch
from torch import nn, FloatTensor, IntTensor

from allennlp.common.detectron import DetectronConfig, pack_images
from allennlp.common.registrable import Registrable


class GridEmbedder(nn.Module, Registrable):
    """
    A `GridEmbedder` takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), and returns a tensor in the format
    (Batch, Features, new_height, new_width).

    For every image, it embeds a patch of the image, and returns the embedding
    of the patch. The size of the image might change during this operation.
    """

    def forward(self, images: FloatTensor, sizes: IntTensor) -> FloatTensor:
        raise NotImplementedError()

    def get_output_dim(self) -> int:
        """
        Returns the output dimension that this `GridEmbedder` uses to represent each
        patch. This is `not` the shape of the returned tensor, but the second dimension
        of that shape.
        """
        raise NotImplementedError

    def get_stride(self) -> int:
        """
        Returns the overall stride of this `GridEmbedder`, which, when combined with the input image
        size, will give you the height and width of the output grid.
        """
        raise NotImplementedError


@GridEmbedder.register("null")
class NullGridEmbedder(GridEmbedder):
    """A `GridEmbedder` that returns the input image as given."""

    def forward(self, images: FloatTensor, sizes: IntTensor) -> FloatTensor:
        return images

    def get_output_dim(self) -> int:
        return 3

    def get_stride(self) -> int:
        return 1


@GridEmbedder.register("detectron_backbone")
class DetectronBackbone(GridEmbedder):
    """Runs an image through a Detectron2 backbone, such as resnet."""

    def __init__(
        self,
        config: DetectronConfig = DetectronConfig.from_flat_parameters(),
    ):
        super().__init__()
        self.config = config
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.config.MODEL.PIXEL_MEAN).view(-1, 1, 1).to(self.config.MODEL.DEVICE),
        )
        self.register_buffer(
            "pixel_std",
            torch.Tensor(self.config.MODEL.PIXEL_STD).view(-1, 1, 1).to(self.config.MODEL.DEVICE),
        )
        self.backbone = self.config.build_backbone()
        if len(self.backbone.output_shape()) != 1:
            raise ValueError(
                "DetectronBackbone currently only supports backbones that produce a single feature"
            )
        self._feature_name = list(self.backbone.output_shape())[0]

    def preprocess(self, images: FloatTensor, sizes: IntTensor) -> FloatTensor:
        # Adapted from https://github.com/facebookresearch/detectron2/blob/
        # 268c90107fba2fea18b1132e5f60532595d771c0/detectron2/modeling/meta_arch/rcnn.py#L224.
        raw_images = [
            (image[:, :height, :width] * 256).byte().to(self.config.MODEL.DEVICE)
            for image, (height, width) in zip(images, sizes)
        ]
        standardized = [(x - self.pixel_mean) / self.pixel_std for x in raw_images]
        return pack_images(standardized, self.backbone.size_divisibility)

    def forward(self, images: FloatTensor, sizes: IntTensor) -> FloatTensor:
        images = self.preprocess(images, sizes)
        result = self.backbone(images)
        assert len(result) == 1
        return result[self._feature_name]

    def get_output_dim(self) -> int:
        return self.backbone.output_shape()[self._feature_name].channels

    def get_stride(self) -> int:
        return self.backbone.output_shape()[self._feature_name].stride
