import itertools
import random
from collections import OrderedDict
from typing import NamedTuple, Optional, List, Tuple

import torch
from torch import nn, FloatTensor, IntTensor, Tensor
import torch.nn.functional as F
import torchvision
import torchvision.ops.boxes as box_ops

from allennlp.common import Registrable


class RegionDetectorOutput(NamedTuple):
    """
    The output type from the forward pass of a `RegionDetector`.
    """

    features: List[Tensor]
    """
    A list of tensors, each with shape `(num_boxes, feature_dim)`.
    """

    boxes: List[Tensor]
    """
    A list of tensors containing the coordinates for each box. Each has shape `(num_boxes, 4)`.
    """

    class_probs: Optional[List[Tensor]] = None
    """
    An optional list of tensors. These tensors can have shape `(num_boxes,)` or
    `(num_boxes, *)` if probabilities for multiple classes are given.
    """

    class_labels: Optional[List[Tensor]] = None
    """
    An optional list of tensors that give the labels corresponding to the `class_probs`
    tensors. This should be non-`None` whenever `class_probs` is, and each tensor
    should have the same shape as the corresponding tensor from `class_probs`.
    """


class RegionDetector(nn.Module, Registrable):
    """
    A `RegionDetector` takes a batch of images, their sizes, and an ordered dictionary
    of image features as input, and finds regions of interest (or "boxes") within those images.

    Those regions of interest are described by three values:

    - `features` (`List[Tensor]`): A feature vector for each region, which is a tensor of shape
      `(num_boxes, feature_dim)`.
    - `boxes` (`List[Tensor]`): The coordinates of each region within the original image, with shape
      `(num_boxes, 4)`.
    - `class_probs` (`Optional[List[Tensor]]`): Class probabilities from some object
      detector that was used to find the regions of interest, with shape `(num_boxes,)`
      or `(num_boxes, *)` if probabilities for more than one class are given.
    - `class_labels` (`Optional[List[Tensor]]`): The labels corresponding to `class_probs`.
      Each tensor in this list has the same shape as the corresponding tensor in `class_probs`.

    """

    def forward(
        self,
        images: FloatTensor,
        sizes: IntTensor,
        image_features: "OrderedDict[str, FloatTensor]",
    ) -> RegionDetectorOutput:
        raise NotImplementedError()


@RegionDetector.register("random")
class RandomRegionDetector(RegionDetector):
    """
    A `RegionDetector` that returns two proposals per image, for testing purposes.  The features for
    the proposal are a random 10-dimensional vector, and the coordinates are the size of the image.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.random = random.Random(seed)

    def _seeded_random_tensor(self, *shape: int, device) -> torch.FloatTensor:
        """PyTorch's random functions can't take a random seed. There is only one global
        random seed in torch, but that's not deterministic enough for us. So we use Python's
        random source to make random tensors."""
        result = torch.zeros(*shape, dtype=torch.float32, device=device)
        for coordinates in itertools.product(*(range(size) for size in result.shape)):
            result[coordinates] = self.random.uniform(-1, 1)
        return result

    def forward(
        self,
        images: FloatTensor,
        sizes: IntTensor,
        image_features: "OrderedDict[str, FloatTensor]",
    ) -> RegionDetectorOutput:
        batch_size, num_features, height, width = images.size()
        features = [
            self._seeded_random_tensor(2, 10, device=images.device) for _ in range(batch_size)
        ]
        boxes = [
            torch.zeros(2, 4, dtype=torch.float32, device=images.device) for _ in range(batch_size)
        ]
        for image_num in range(batch_size):
            boxes[image_num][0, 2] = sizes[image_num, 0]
            boxes[image_num][0, 3] = sizes[image_num, 1]
            boxes[image_num][1, 2] = sizes[image_num, 0]
            boxes[image_num][1, 3] = sizes[image_num, 1]
        return RegionDetectorOutput(features, boxes)


@RegionDetector.register("faster_rcnn")
class FasterRcnnRegionDetector(RegionDetector):
    """
    A [Faster R-CNN](https://arxiv.org/abs/1506.01497) pretrained region detector.

    Unless you really know what you're doing, this should be used with the image
    features created from the `ResnetBackbone` `GridEmbedder` and on images loaded
    using the `TorchImageLoader` with the default settings.


    !!! Note
        This module does not have any trainable parameters by default.
        All pretrained weights are frozen.

    # Parameters

    box_score_thresh : `float`, optional (default = `0.05`)
        During inference, only proposal boxes / regions with a label classification score
        greater than `box_score_thresh` will be returned.

    box_nms_thresh : `float`, optional (default = `0.5`)
        During inference, non-maximum suppression (NMS) will applied to groups of boxes
        that share a common label.

        NMS iteratively removes lower scoring boxes which have an intersection-over-union (IoU)
        greater than `box_nms_thresh` with another higher scoring box.

    max_boxes_per_image : `int`, optional (default = `100`)
        During inference, at most `max_boxes_per_image` boxes will be returned. The
        number of boxes returned will vary by image and will often be lower
        than `max_boxes_per_image` depending on the values of `box_score_thresh`
        and `box_nms_thresh`.
    """

    def __init__(
        self,
        *,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        max_boxes_per_image: int = 100,
    ):
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=max_boxes_per_image,
        )
        # Don't need this since the features will be calculated elsewhere.
        del self.detector.backbone
        # Freeze all weights.
        for parameter in self.detector.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        images: FloatTensor,
        sizes: IntTensor,
        image_features: "OrderedDict[str, FloatTensor]",
    ) -> RegionDetectorOutput:
        """
        Extract regions and region features from the given images.

        In most cases `image_features` should come directly from the `ResnetBackbone`
        `GridEmbedder`. The `images` themselves should be standardized and resized
        using the default settings for the `TorchImageLoader`.
        """
        if self.training:
            raise RuntimeError(
                "FasterRcnnRegionDetector can not be used for training at the moment"
            )

        # Adapted from https://github.com/pytorch/vision/blob/
        # 4521f6d152875974e317fa247a633e9ad1ea05c8/torchvision/models/detection/generalized_rcnn.py#L45
        # We re-implement essentially the same forward eval pass except that we
        # skip calling the backbone since we already have the `image_features`,
        # and we also unpack the call to `roi_heads` so that we can keep the `box_features`
        # that are created here:
        # https://github.com/pytorch/vision/blob/
        # 4521f6d152875974e317fa247a633e9ad1ea05c8/torchvision/models/detection/roi_heads.py#L752-L753

        image_shapes: List[Tuple[int, int]] = list((int(h), int(w)) for (h, w) in sizes)
        image_list = torchvision.models.detection.image_list.ImageList(images, image_shapes)

        # `proposals` is a list of tensors, one tensor per image, each representing a
        # fixed number of proposed regions/boxes.
        # shape (proposals[i]): (proposals_per_image, 4)
        proposals: List[Tensor]
        proposals, _ = self.detector.rpn(image_list, image_features)

        # shape: (batch_size * proposals_per_image, *)
        box_features = self.detector.roi_heads.box_roi_pool(image_features, proposals, image_shapes)

        # shape: (batch_size * proposals_per_image, *)
        box_features = self.detector.roi_heads.box_head(box_features)

        # shape (class_logits): (batch_size * proposals_per_image, num_classes)
        # shape (box_regression): (batch_size * proposals_per_image, regression_output_size)
        class_logits, box_regression = self.detector.roi_heads.box_predictor(box_features)

        # This step filters down the `proposals` to only detections that reach
        # a certain threshold.
        # Each of these is a list of tensors, one for each image in the batch.
        # shape (boxes[i]): (num_predicted_boxes, 4)
        # shape (features[i]): (num_predicted_boxes, feature_size)
        # shape (scores[i]): (num_predicted_classes,)
        # shape (labels[i]): (num_predicted_classes,)
        boxes, features, scores, labels = self._postprocess_detections(
            class_logits, box_features, box_regression, proposals, image_shapes
        )

        return RegionDetectorOutput(features, boxes, scores, labels)

    def _postprocess_detections(
        self,
        class_logits: Tensor,
        box_features: Tensor,
        box_regression: Tensor,
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """
        Adapted from https://github.com/pytorch/vision/blob/
        4521f6d152875974e317fa247a633e9ad1ea05c8/torchvision/models/detection/roi_heads.py#L664.

        The only reason we have to re-implement this method is so we can pull out the box
        features that we want.
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        # shape: (batch_size * boxes_per_image, num_classes, 4)
        pred_boxes = self.detector.roi_heads.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        features_list = box_features.split(boxes_per_image, dim=0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_features = []
        all_scores = []
        all_labels = []
        for boxes, features, scores, image_shape in zip(
            pred_boxes_list, features_list, pred_scores_list, image_shapes
        ):
            # shape: (boxes_per_image, num_classes, 4)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # shape: (boxes_per_image, num_classes, feature_size)
            features = features.unsqueeze(1).expand(boxes.shape[0], boxes.shape[1], -1)

            # create labels for each prediction
            # shape: (num_classes,)
            labels = torch.arange(num_classes, device=device)
            # shape: (boxes_per_image, num_classes,)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            # shape: (boxes_per_image, num_classes - 1, 4)
            boxes = boxes[:, 1:]
            # shape: (boxes_per_image, num_classes, feature_size)
            features = features[:, 1:]
            # shape: (boxes_per_image, num_classes - 1,)
            scores = scores[:, 1:]
            # shape: (boxes_per_image, num_classes - 1,)
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            # shape: (boxes_per_image * (num_classes - 1), 4)
            boxes = boxes.reshape(-1, 4)
            # shape: (boxes_per_image * (num_classes - 1), feature_size)
            features = features.reshape(boxes.shape[0], -1)
            # shape: (boxes_per_image * (num_classes - 1),)
            scores = scores.reshape(-1)
            # shape: (boxes_per_image * (num_classes - 1),)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.detector.roi_heads.score_thresh)[0]
            boxes, features, scores, labels = (
                boxes[inds],
                features[inds],
                scores[inds],
                labels[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, features, scores, labels = (
                boxes[keep],
                features[keep],
                scores[keep],
                labels[keep],
            )

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.detector.roi_heads.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detector.roi_heads.detections_per_img]
            boxes, features, scores, labels = (
                boxes[keep],
                features[keep],
                scores[keep],
                labels[keep],
            )

            all_boxes.append(boxes)
            all_features.append(features)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_features, all_scores, all_labels
