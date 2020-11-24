from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, FloatTensor, IntTensor

from allennlp.common.detectron import DetectronConfig, ImageList
from allennlp.common.lazy import Lazy
from allennlp.common.registrable import Registrable


class RegionDetector(nn.Module, Registrable):
    """
    A `RegionDetector` takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), and finds regions of interest (or "boxes") within those images.

    Those regions of interest are described by three values:
    - A feature vector for each region, which is a tensor of shape (Batch, NumBoxes, FeatureDim)
    - The coordinates of each region within the original image, with shape (Batch, NumBoxes, 4)
    - (Optionally) Class probabilities from some object detector that was used to find the regions
      of interest, with shape (Batch, NumBoxes, NumClasses).

    Because the class probabilities are an optional return value, we return these tensors in a
    dictionary, instead of a tuple (so you don't have a confusing check on the tuple size).  The
    keys are "coordinates", "features", and "class_probs".
    """

    def forward(
        self, raw_images: FloatTensor, image_sizes: IntTensor, featurized_images: FloatTensor
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()


@RegionDetector.register("random")
class RandomRegionDetector(RegionDetector):
    """
    A `RegionDetector` that returns two proposals per image, for testing purposes.  The features for
    the proposal are a random 10-dimensional vector, and the coordinates are the size of the image.
    """

    def forward(
        self, raw_images: FloatTensor, image_sizes: IntTensor, featurized_images: FloatTensor
    ) -> Dict[str, torch.Tensor]:
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
        return {"features": features, "coordinates": coordinates}


@RegionDetector.register("faster_rcnn")
class FasterRcnnRegionDetector(RegionDetector):
    """
    Faster R-CNN (https://arxiv.org/abs/1506.01497) with ResNet backbone.
    Based on detectron2 v0.2
    """

    def __init__(
        self,
        config: Lazy[DetectronConfig] = Lazy(DetectronConfig.from_flat_parameters),
        detections_per_image: int = 36,
    ):
        super().__init__()
        self.config: DetectronConfig = config.construct()
        self.detections_per_image = detections_per_image
        self.register_buffer(
            "pixel_mean", torch.Tensor(self.config.cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(self.config.cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        self._model_object = None

    def preprocess(self, images: FloatTensor, sizes: IntTensor) -> ImageList:
        # Adapted from https://github.com/facebookresearch/detectron2/blob/
        # 268c90107fba2fea18b1132e5f60532595d771c0/detectron2/modeling/meta_arch/rcnn.py#L224.
        raw_images = [
            (image[:, :height, :width] * 256).byte().to(self.flat_parameters.device)
            for image, (height, width) in zip(images, sizes)
        ]
        standardized = [(x - self.pixel_mean) / self.pixel_std for x in raw_images]
        return ImageList.from_tensors(standardized, self._model.backbone.size_divisibility)

    @property
    def _model(self):
        if self._model_object is None:
            self._model_object = self.config.build_model()
        return self._model_object

    def forward(
        self, raw_images: FloatTensor, image_sizes: IntTensor, featurized_images: FloatTensor
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(image_sizes)

        image_list = self.preprocess(raw_images, image_sizes)

        # RPN
        assert len(self._model.proposal_generator.in_features) == 1
        featurized_images_in_dict = {
            self._model.proposal_generator.in_features[0]: featurized_images
        }

        proposals, _ = self._model.proposal_generator(image_list, featurized_images_in_dict, None)

        # this will concatenate the pooled_features from different images.
        _, pooled_features = self._model.roi_heads.get_roi_features(
            featurized_images_in_dict, proposals
        )

        predictions = self._model.roi_heads.box_predictor(pooled_features)

        # class probability
        cls_probs = F.softmax(predictions[0], dim=-1)
        cls_probs = cls_probs[:, :-1]  # background is last

        predictions, r_indices = self._model.roi_heads.box_predictor.inference(
            predictions, proposals
        )

        batch_coordinates = []
        batch_features = []
        batch_probs = []
        batch_num_detections = torch.zeros(
            batch_size, device=image_list.tensor.device, dtype=torch.int16
        )
        num_classes = cls_probs.size(-1)

        proposal_start = 0
        for image_num, image_proposal_indices in enumerate(r_indices):
            # "proposals" are the things that the proposal generator above thought might be useful
            # boxes.  "detections" are proposals that made it passed the ROI feature and box
            # predictor step.  We keep only the "detections", but we the features and predictions
            # come from before we decided which things were "detections".
            num_detections = len(image_proposal_indices)
            num_image_proposals = len(proposals[image_num])
            batch_num_detections[image_num] = num_detections

            coordinates = proposals[image_num].proposal_boxes.tensor[image_proposal_indices]

            # Detectron puts all of the features and predictions into a squashed tensor of shape
            # (total_num_proposals, feature_dim | num_classes).  This means that we have to iterate
            # over these tensors in a funny way, keeping track of where we are in the tensor,
            # instead of just doing a simple `.view()`.
            image_features = torch.narrow(pooled_features, 0, proposal_start, num_image_proposals)
            features = image_features[image_proposal_indices]
            image_probs = torch.narrow(cls_probs, 0, proposal_start, num_image_proposals)
            probs = image_probs[image_proposal_indices]

            proposal_start += num_image_proposals
            if num_detections < self.detections_per_image:
                num_to_add = self.detections_per_image - num_detections

                coords_padding = coordinates.new_zeros(size=(num_to_add, 4))
                coordinates = torch.cat([coordinates, coords_padding], dim=0)

                probs_padding = probs.new_zeros(size=(num_to_add, num_classes))
                probs = torch.cat([probs, probs_padding], dim=0)

                num_features = features.size(-1)
                features_padding = features.new_zeros(size=(num_to_add, num_features))
                features = torch.cat([features, features_padding], dim=0)

            batch_coordinates.append(coordinates)
            batch_features.append(features)
            batch_probs.append(probs)

        features_tensor = torch.stack(batch_features, dim=0)
        coordinates = torch.stack(batch_coordinates, dim=0)
        probs_tensor = torch.stack(batch_probs, dim=0)
        return {
            "features": features_tensor,
            "coordinates": coordinates,
            "class_probs": probs_tensor,
            "num_regions": batch_num_detections,
        }
