import torch
from torch import nn, FloatTensor, IntTensor
from typing import List

from allennlp.common.registrable import Registrable


class Image2ImageModule(nn.Module, Registrable):
    """
    An Image2ImageModule takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), and returns a tensor in the same format, after
    applying some transformation on the images.
    """

    def forward(self, images: FloatTensor, sizes: IntTensor):
        raise NotImplementedError()


@Image2ImageModule.register("null")
class NullImage2ImageModule(Image2ImageModule):
    """An `Image2ImageModule` that returns the original image unchanged."""

    def forward(self, images: FloatTensor, sizes: IntTensor):
        return images


@Image2ImageModule.register("normalize")
class NormalizeImage(Image2ImageModule):
    """
    Normalizes an image by subtracting the mean and dividing by the
    standard deviation, separately for each channel.
    """

    def __init__(self, means: List[float], stds: List[float]):
        super().__init__()
        assert len(means) == len(stds)
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def forward(self, images: FloatTensor, sizes: IntTensor):
        assert images.size(1) == self.means.size(0)
        self.means = self.means.to(images.device)
        self.stds = self.stds.to(images.device)
        images = images.transpose(1, -1)
        images = images - self.means
        images = images / self.stds
        return images.transpose(-1, 1)


@Image2ImageModule.register("resnet_backbone")
class ResnetBackbone(Image2ImageModule):
    """Runs an image through resnet, as implemented by Detectron."""

    def __init__(self,
        meta_architecture: str = "GeneralizedRCNN",
        device: str = "cpu",
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

        flat_parameters = detectron.DetectronFlatParameters(
            max_attr_per_ins=max_attr_per_ins,
            device=device,
            weights=weights,
            meta_architecture=meta_architecture,
            attribute_on=attribute_on,
            stride_in_1x1=stride_in_1x1,
            num_groups=num_groups,
            width_per_group=width_per_group,
            depth=depth)

        pipeline = detectron.get_pipeline_from_flat_parameters(flat_parameters, make_copy=False)
        self.backbone = pipeline.model.backbone

    def forward(self, images: FloatTensor, sizes: IntTensor) -> FloatTensor:
        result = self.backbone(images)
        assert len(result) == 1
        return next(iter(result.values()))
