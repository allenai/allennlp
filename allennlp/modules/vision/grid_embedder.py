from collections import OrderedDict
from typing import Tuple

from torch import nn, FloatTensor, IntTensor
import torchvision

from allennlp.common.registrable import Registrable


class GridEmbedder(nn.Module, Registrable):
    """
    A `GridEmbedder` takes a batch of images as a tensor with shape
    `(batch_size, color_channels, height, width)`, and returns an ordered dictionary
    of tensors with shape `(batch_size, *)`, each representing a specific feature.
    """

    def forward(self, images: FloatTensor, sizes: IntTensor) -> "OrderedDict[str, FloatTensor]":
        raise NotImplementedError()

    def get_feature_names(self) -> Tuple[str, ...]:
        """
        Returns the feature names, in order, i.e. the keys of the ordered output
        dictionary from `.forward()`.
        """
        raise NotImplementedError()


@GridEmbedder.register("null")
class NullGridEmbedder(GridEmbedder):
    """
    A `GridEmbedder` that returns the input image as given.
    """

    def forward(self, images: FloatTensor, sizes: IntTensor) -> "OrderedDict[str, FloatTensor]":
        out = OrderedDict()
        out["0"] = images
        return out

    def get_feature_names(self) -> Tuple[str, ...]:
        return ("0",)


@GridEmbedder.register("resnet_backbone")
class ResnetBackbone(GridEmbedder):
    """
    Runs an image through [ResNet](https://api.semanticscholar.org/CorpusID:206594692),
    as implemented by [torchvision](https://pytorch.org/docs/stable/torchvision/models.html).
    """

    def __init__(self) -> None:
        super().__init__()
        detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.backbone = detection_model.backbone
        # Don't need the rest of this.
        del detection_model
        self.feature_names = tuple(
            [
                self.backbone.body.return_layers[key]
                for key in self.backbone.body.keys()
                if key in self.backbone.body.return_layers
            ]
            + ["pool"]
        )

    def forward(self, images: FloatTensor, sizes: IntTensor) -> "OrderedDict[str, FloatTensor]":
        return self.backbone(images)

    def get_feature_names(self) -> Tuple[str, ...]:
        return self.feature_names
