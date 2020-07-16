from os import PathLike
from typing import NamedTuple, Tuple, Union, List, Dict

import torch
import torch.nn.functional as F
from detectron2.config import CfgNode
from detectron2.modeling import postprocessing
from torch import Tensor

from allennlp.data import Field


class ImageWithSize(NamedTuple):
    image: Union[Tensor, str, PathLike]
    size: Tuple[int, int]


SupportedImageFormat = Union[ImageWithSize, Tensor, dict, str, PathLike]


class DetectronProcessor:
    def __init__(self, cfg: CfgNode):
        from detectron2.data import DatasetMapper

        self.cfg = cfg
        self.mapper = DatasetMapper(cfg)
        from detectron2.modeling import build_model

        self.model = build_model(cfg)
        from detectron2.checkpoint import DetectionCheckpointer

        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, images: Union[SupportedImageFormat, List[SupportedImageFormat]]) -> Union[Dict[str, Field], List[Dict[str, Field]]]:
        # handle the single-image case
        if not isinstance(images, list):
            return self.__call__([images])[0]

        images = [self._to_model_input(i) for i in images]
        images = self.model.preprocess_image(images)
        features = self.model.backbone(images.tensor)

        # for region features
        if self.cfg.MODEL.FEATURE_TYPE == "region":

            # assume we didn't provide the proposals for now
            proposals, _ = self.model.proposal_generator(images, features, None)
            _, pooled_features = self.model.roi_heads.get_roi_features(features, proposals)

            predictions = self.model.roi_heads.box_predictor(pooled_features)
            cls_probs = F.softmax(predictions[0], dim=-1)
            cls_probs = cls_probs[:, :-1]  # background is last
            predictions, r_indices = self.model.roi_heads.box_predictor.inference(
                predictions, proposals
            )
            # postprocess
            pooled_features = pooled_features[r_indices]
        elif self.cfg.MODEL.FEATURE_TYPE == "grid":
            # for grid features
            pooled_features = self.model.roi_heads.get_conv5_features(features)
        else:
            raise ValueError("Unsupported MODEL.FEATURE_TYPE")

        image_fields = []
        from allennlp.data.fields.tensor_field import ArrayField

        # comment this for now:

        # for i, image in enumerate(images):
        #     fields = {}
        #     if "instances" in image:
        #         instances = image["instances"]
        #         if instances.has("pred_boxes"):
        #             fields["instances/pred_boxes"] = ArrayField(
        #                 instances.get("pred_boxes").tensor, padding_value=-1
        #             )
        #         if instances.has("scores"):
        #             fields["instances/scores"] = ArrayField(instances.get("scores"))
        #         if instances.has("pred_classes"):
        #             fields["instances/pred_classes"] = ArrayField(
        #                 instances.get("pred_classes"), padding_value=-1
        #             )
        #         if instances.has("pred_masks"):
        #             fields["instances/pred_masks"] = ArrayField(
        #                 instances.get("pred_masks"), padding_value=False
        #             )
        #         if instances.has("pred_keypoints"):
        #             fields["instances/pred_keypoints"] = ArrayField(
        #                 instances.get("pred_keypoints"), padding_value=-1
        #             )
        #     if "sem_seg" in image:
        #         fields["sem_seg"] = ArrayField(image["sem_seg"], padding_value=0.0)
        #     if "proposals" in image:
        #         instances = image["proposals"]
        #         if instances.has("proposal_boxes"):
        #             fields["proposals/proposal_boxes"] = ArrayField(
        #                 instances.get("proposal_boxes").tensor, padding_value=-1
        #             )
        #         if instances.has("objectness_logits"):
        #             fields["proposals/objectness_logits"] = ArrayField(
        #                 instances.get("objectness_logits")
        #             )
        #     if "panoptic_seg" in image:
        #         segment_ids, dicts = image["panoptic_seg"]
        #         fields["panoptic_seg"] = ArrayField(segment_ids, padding_value=-1)

        #         ids = torch.tensor([d["id"] for d in dicts], dtype=torch.int32)
        #         isthings = torch.tensor([d["isthing"] for d in dicts], dtype=torch.bool)
        #         category_ids = torch.tensor([d["category_id"] for d in dicts], dtype=torch.int32)
        #         fields["panoptic_seg/ids"] = ArrayField(ids, padding_value=-1)
        #         fields["panoptic_seg/isthings"] = ArrayField(isthings, padding_value=False)
        #         fields["panoptic_seg/category_ids"] = ArrayField(category_ids, padding_value=-1)

        #     image_fields.append(fields)
        return pooled_features
        # return image_fields, pooled_features

    def _to_model_input(self, image: SupportedImageFormat) -> dict:
        if isinstance(image, ImageWithSize):
            if isinstance(image.image, PathLike):
                image.image = str(image.image)
            image_dict = {"height": image.size[0], "width": image.size[1]}
            if isinstance(image.image, str):
                image_dict["file_name"] = image.image
            elif isinstance(image.image, Tensor):
                image_dict["image"] = image.image
            else:
                raise ValueError("`image` is not in a recognized format.")
            image = image_dict
        else:
            if isinstance(image, PathLike):
                image = str(image)
            if isinstance(image, str):
                image = {"file_name": image}
        assert isinstance(image, dict)
        if "image" not in image:
            image = self.mapper(image)
        assert isinstance(image["image"], Tensor)
        return image
