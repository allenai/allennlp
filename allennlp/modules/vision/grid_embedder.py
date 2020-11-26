from typing import Union

import torch
from torch import nn, FloatTensor, IntTensor

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


@GridEmbedder.register("resnet_backbone")
class ResnetBackbone(GridEmbedder):
    """Runs an image through resnet, as implemented by Detectron."""

    def __init__(
        self,
        meta_architecture: str = "GeneralizedRCNN",
        device: Union[str, int, torch.device] = "cpu",
        weights: str = "RCNN-X152-C4-2020-07-18",
        attribute_on: bool = True,  # not in detectron2 default config
        max_attr_per_ins: int = 16,  # not in detectron2 default config
        stride_in_1x1: bool = False,  # different from default (True)
        num_groups: int = 32,  # different from default (1)
        width_per_group: int = 8,  # different from default (64)
        depth: int = 152,  # different from default (50)
    ):
        super().__init__()
        from allennlp.common import detectron

        self.flat_parameters = detectron.DetectronFlatParameters(
            max_attr_per_ins=max_attr_per_ins,
            device=device,
            weights=weights,
            meta_architecture=meta_architecture,
            attribute_on=attribute_on,
            stride_in_1x1=stride_in_1x1,
            num_groups=num_groups,
            width_per_group=width_per_group,
            depth=depth,
        )
        self._pipeline_object = None

    @property
    def _pipeline(self):
        if self._pipeline_object is None:
            from allennlp.common import detectron

            self._pipeline_object = detectron.get_pipeline_from_flat_parameters(
                self.flat_parameters, make_copy=False
            )
        return self._pipeline_object

    @property
    def preprocessor(self):
        return self._pipeline.model.preprocess_image

    @property
    def backbone(self):
        return self._pipeline.model.backbone

    def forward(self, images: FloatTensor, sizes: IntTensor) -> FloatTensor:
        images = [
            {"image": (image[:, :height, :width] * 256).byte(), "height": height, "width": width}
            for image, (height, width) in zip(images, sizes)
        ]
        images = self.preprocessor(images)  # This returns tensors on the correct device.
        result = self.backbone(images.tensor)
        assert len(result) == 1
        return next(iter(result.values()))

    def get_output_dim(self) -> int:
        return self.backbone.output_shape()["res4"].channels

    def get_stride(self) -> int:
        return self.backbone.output_shape()["res4"].stride

    def to(self, device):
        if isinstance(device, int) or isinstance(device, torch.device):
            if self._pipeline_object is not None:
                self._pipeline_object.model.to(device)
            if isinstance(device, torch.device):
                device = device.index
            self.flat_parameters = self.flat_parameters._replace(device=device)
            return self
        else:
            return super().to(device)
