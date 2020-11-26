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

    # Parameters

    min_size : `int`, optional (default = `800`)
        Images smaller than this will be resized up to `min_size` before feeding into the backbone.
    max_size : `int`, optional (default = `1333`)
        Images larger than this will be resized down to `max_size` before feeding into the backbone.
    image_mean : `Tuple[float, float, float]`, optional (default = `(0.485, 0.456, 0.406)`)
        Mean values for image normalization.
    image_std : `Tuple[float, float, float]`, optional (default = `(0.229, 0.224, 0.225)`)
        Standard deviation for image normalization.
    """

    def __init__(
        self,
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            "resnet50", pretrained=True, trainable_layers=0
        )
        self.backbone.eval()
        self.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std
        )
        self.feature_names = tuple(
            [
                self.backbone.body.return_layers[key]
                for key in self.backbone.body.keys()
                if key in self.backbone.body.return_layers
            ]
            + ["pool"]
        )

    def forward(self, images: FloatTensor, sizes: IntTensor) -> "OrderedDict[str, FloatTensor]":
        # self.transform takes a list of single images.
        # shape (image_list[i]): (color_channels, height, width)
        image_list = list(images)

        transformed_list, _ = self.transform(list(images))

        # shape: (batch_size, color_channels, new_height, new_width)
        transformed = transformed_list.tensors

        return self.backbone(transformed)

    def get_feature_names(self) -> Tuple[str, ...]:
        return self.feature_names
