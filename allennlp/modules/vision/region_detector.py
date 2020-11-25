from typing import Dict, NamedTuple

import torch
from torch import nn, FloatTensor, IntTensor

from allennlp.common.detectron import DetectronConfig, ImageList
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


@RegionDetector.register("detectron_rcnn")
class DetectronRcnnRegionDetector(RegionDetector):
    def __init__(
        self,
        config: DetectronConfig,
        detections_per_image: int = 36,
    ):
        super().__init__()

        if config.MODEL.META_ARCHITECTURE != "GeneralizedRCNN":
            raise ValueError("Expected a GeneralizedRCNN detectron meta architecture")

        self.config = config
        self.detections_per_image = detections_per_image
        model = self.config.build_model()

        # HACK: We don't actually need to backbone of the model, at least not it's weights,
        # so to save memory we remove it from the model and patch it with a small class
        # that will work in its place.
        class BackbonePatch(NamedTuple):
            size_divisibility: int = 0

        backbone_patch = BackbonePatch(size_divisibility=model.backbone.size_divisibility)
        del model.backbone
        model.backbone = backbone_patch

        self.model = model

        if len(self.model.proposal_generator.in_features) != 1:
            raise ValueError(
                "DetectronRcnnRegionDetector currently only supports detectron proposal generators "
                "that take a single input feature"
            )

        self._in_feature_name = self.model.proposal_generator.in_features[0]

    def preprocess(self, images: FloatTensor, sizes: IntTensor) -> ImageList:
        # Adapted from https://github.com/facebookresearch/detectron2/blob/
        # 268c90107fba2fea18b1132e5f60532595d771c0/detectron2/modeling/meta_arch/rcnn.py#L224.
        raw_images = [
            (image[:, :height, :width] * 256).byte().to(self.config.MODEL.DEVICE)
            for image, (height, width) in zip(images, sizes)
        ]
        standardized = [(x - self.pixel_mean) / self.pixel_std for x in raw_images]
        return ImageList.from_tensors(standardized, self._size_divisibility)

    def forward(
        self, raw_images: FloatTensor, image_sizes: IntTensor, featurized_images: FloatTensor
    ) -> Dict[str, torch.Tensor]:
        # Adapted from https://github.com/facebookresearch/detectron2/blob/
        # 786875e03a3d88a06e4b3ff9cfd5120b0f04e14c/detectron2/modeling/meta_arch/rcnn.py#L177.

        # Detectron expects the input to look like this:
        batched_inputs = [
            {"image": (image[:, :height, :width] * 256).byte().to(self.config.MODEL.DEVICE)}
            for image, (height, width) in zip(raw_images, image_sizes)
        ]

        # Run through the same steps as in detectron2 GeneralizedRCNN.inference(),
        # except we skip the backbone since we already have image features.
        images = self.model.preprocess_image(batched_inputs)
        features = {self._in_feature_name: featurized_images}
        proposals, _ = self.model.proposal_generator(images, features, None)
        results, _ = self.model.roi_heads(images, features, proposals, None)
        results = self.model._postprocess(results, batched_inputs, images.image_sizes)

        # Now we just to transform `results` to match the format our RegionDetector base
        # class specifies.
        # `results` is a list of length batch size.

        return results
