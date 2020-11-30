from typing import Dict, List, Tuple, Union, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn, FloatTensor, IntTensor

from allennlp.common import Registrable


class RegionDetectorOutput(NamedTuple):
    features: torch.Tensor
    coordinates: torch.Tensor
    class_probs: Optional[torch.Tensor] = None


class RegionDetector(nn.Module, Registrable):
    """
    A `RegionDetector` takes a batch of images, their sizes, and an ordered dictionary
    of image features as input, and finds regions of interest (or "boxes") within those images.

    Those regions of interest are described by three values:

    - `features`: A feature vector for each region, which is a tensor of shape
      `(batch_size, num_boxes, feature_dim)`.
    - `coordinates`: The coordinates of each region within the original image, with shape
      `(batch_size, num_boxes, 4)`.
    - `class_probs` (optional): Class probabilities from some object detector that was
      used to find the regions of interest, with shape `(batch_size, num_boxes, num_classes)`.

    """

    def forward(
        self,
        raw_images: FloatTensor,
        image_sizes: IntTensor,
        featurized_images: "OrderedDict[str, FloatTensor]",
    ) -> RegionDetectorOutput:
        raise NotImplementedError()


@RegionDetector.register("random")
class RandomRegionDetector(RegionDetector):
    """
    A `RegionDetector` that returns two proposals per image, for testing purposes.  The features for
    the proposal are a random 10-dimensional vector, and the coordinates are the size of the image.
    """

    def forward(
        self,
        raw_images: FloatTensor,
        image_sizes: IntTensor,
        featurized_images: "OrderedDict[str, FloatTensor]",
    ) -> RegionDetectorOutput:
        batch_size, num_features, height, width = raw_images.size()
        features = torch.rand(batch_size, 2, 10, dtype=featurized_images.dtype).to(
            raw_images.device
        )
        coordinates = torch.zeros(batch_size, 2, 4, dtype=torch.float32).to(raw_images.device)
        for image_num in range(batch_size):
            coordinates[image_num, 0, 2] = image_sizes[image_num, 0]
            coordinates[image_num, 0, 3] = image_sizes[image_num, 1]
            coordinates[image_num, 1, 2] = image_sizes[image_num, 0]
            coordinates[image_num, 1, 3] = image_sizes[image_num, 1]
        return RegionDetectorOutput(features, coordinates)


@RegionDetector.register("faster_rcnn")
class FasterRcnnRegionDetector(RegionDetector):
    """
    [Faster R-CNN](https://arxiv.org/abs/1506.01497) region detector.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        raw_images: FloatTensor,
        image_sizes: IntTensor,
        featurized_images: "OrderedDict[str, FloatTensor]",
    ) -> RegionDetectorOutput:
        pass
