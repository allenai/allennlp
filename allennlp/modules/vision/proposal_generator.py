from os import PathLike
from typing import NamedTuple, Tuple, Union, List, Dict

import torch
import torch.nn.functional as F
from torch import nn, FloatTensor
from torch import Tensor

from allennlp.common.registrable import Registrable

class ImageWithSize(NamedTuple):
    image: Union[Tensor, str, PathLike]
    size: Tuple[int, int]

SupportedImageFormat = Union[ImageWithSize, Tensor, dict, str, PathLike]


class ProposalGenerator(nn.Module, Registrable):
    """
    A `ProposalGenerator` takes a batch of images as a tensor with the dimensions
    (Batch, Color, Height, Width), and returns a tensor in the format (Batch, #Boxes, 4).
    In other words, for every image, it returns a number of proposed boxes, identified by
    their four coordinates `(x1, y2, x2, y2)`. Coordinates are expected to be between 0
    and 1. Negative coordinates are interpreted as padding.
    """

    def forward(self, images: FloatTensor):
        raise NotImplementedError()


# A `ProposalGenerator` takes a batch of images as a tensor with the dimensions
# (Batch, Color, Height, Width), and returns bounding box information below:
#     1) box coordinates(Batch,  # Boxes, 4)
#     2) box classification probability(Batch,  # Boxes, #classes)
#     3) ROI - pooled box features(Batch,  # Boxes, #features)

from allennlp.common.file_utils import cached_path
import tarfile
from filelock import FileLock
import shutil
import os

def _load_detectron_checkpoint(checkpoint_name="RCNN-X152-C4-2020-07-18"):
    """
    Download and cache a tarfile containing detectron checkpoint (_weights.th)
    It might be better to replace this with native utility functions like load_archive (https://github.com/allenai/allennlp/blob/master/allennlp/models/archival.py#L132) later,
    where the tarfile includes 1) config.json 2) _weights.th
    """
    gs_url = "https://storage.googleapis.com/allennlp-public-models"
    detectron_checkpoint_url = f'{gs_url}/{checkpoint_name}.tar.gz'
    extraction_path = cached_path(detectron_checkpoint_url, extract_archive=True)

    folder_name = detectron_checkpoint_url.split('/')[-1].split('.')[0]  # ex) RCNN-X152-C4-2020-07-18
    weights_path = f'{extraction_path}/{folder_name}/_weights.th'

    return weights_path


@ProposalGenerator.register("Faster-RCNN")
class FasterRCNNProposalGenerator(ProposalGenerator):
    """
    Faster R-CNN (https://arxiv.org/abs/1506.01497) with ResNet backbone.
    Based on detectron2 v0.2
    """
    def __init__(
        self,
        meta_architecture = "GeneralizedRCNN",
        device: str = "cpu",
        weights: str = "RCNN-X152-C4-2020-07-18",

        attribute_on: bool = True, # not in detectron2 default config
        max_attr_per_ins: int = 16,  # not in detectron2 default config

        # Resnet (ResNeXt) backbone
        stride_in_1x1: bool = False,  # different from default (True)
        num_groups: int = 32,  # different from default (1)
        width_per_group: int = 8,  # different from default (64)
        depth: int = 152,  # different from default (50)

        # RPN
        rpn_head_name: str = "StandardRPNHead",
        rpn_in_features: List[str] = ["res4"],
        rpn_boundary_thresh: int = 0,
        rpn_iou_thresholds: List[float] = [0.3, 0.7],
        rpn_iou_labels: List[int] = [0, -1, 1],
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        rpn_bbox_reg_loss_type: str = 'smooth_l1',
        rpn_bbox_reg_loss_weight: float = 1.0,  # different from default (-1)
        rpn_bbox_reg_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
        rpn_smooth_l1_beta: float = 0.1111,  # different from default (0.0)
        rpn_loss_weight: float = 1.0,
        rpn_pre_nms_topk_train: int = 12000,
        rpn_pre_nms_topk_test: int = 6000,
        rpn_post_nms_topk_train: int = 2000,
        rpn_post_nms_topk_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_bbox_loss_weight: float = 1.0,  # not in detectron2 default config

        # ROI head
        roi_head_name: str = "AttributeRes5ROIHeads",  # different from default (Res5ROIHeads)
        roi_head_num_classes: int = 1600,  # different from default (80)
        roi_head_in_features: List[str] = ["res4"],
        roi_head_iou_thresholds: List[float] = [0.5],
        roi_head_iou_labels: List[int] = [0, 1],
        roi_head_batch_size_per_image: int = 512,
        roi_head_positive_fraction: float = 0.25,
        roi_head_score_thresh_test: float = 0.05,
        roi_head_nms_thresh_test: float = 0.5,
        roi_head_proposal_append_gt: bool = True,

        # ROI Box head
        box_head_name: str = "FastRCNNConvFCHead",  # different from default ("")
        box_head_bbox_reg_loss_type: str = "smooth_l1",
        box_head_bbox_reg_loss_weight: float = 1.0,
        box_head_bbox_reg_weights: Tuple[float, ...] = (10.0, 10.0, 5.0, 5.0),
        box_head_smooth_l1_beta: float = 1.0,  # different from default (0.0)
        box_head_pooler_resolution: int = 7,  # different from default (14)
        box_head_pooler_sampling_ratio: int = 2,  # different from default (0)
        box_head_pooler_type: str = "ROIAlignV2",
        box_head_num_fc: int = 2,  # different from default (0)
        box_head_fc_dim: int = 1024,
        box_head_num_conv: int = 0,
        box_head_conv_dim: int = 256,
        box_head_norm: str = "",
        box_head_cls_agnostic_bbox_reg: bool = False,
        box_head_train_on_pred_boxes: bool = False,
        box_head_bbox_loss_weight: float = 1.0,  # not in detectron2 default config

        # Box attribute classification head (not in detectron2 default config)
        attribute_head_obj_embed_dim: int = 256,
        attribute_head_fc_dim: int = 512,
        attribute_head_loss_weight: float = 0.2,
        attribute_head_num_classes: int = 400

        ):
        super().__init__()

        if weights:
            weights = _load_detectron_checkpoint(weights)

        overrides = {
            "VERSION": 2,
            "INPUT": {
                "MAX_ATTR_PER_INS": max_attr_per_ins,
            },
            "MODEL": {
                "DEVICE": device,
                "WEIGHTS": weights,
                "META_ARCHITECTURE": meta_architecture,
                "ATTRIBUTE_ON": attribute_on,
                "PROPOSAL_GENERATOR": {
                    "NAME": "RPN",
                },
                "RESNETS": {
                    "STRIDE_IN_1X1": stride_in_1x1,
                    "NUM_GROUPS": num_groups,
                    "WIDTH_PER_GROUP": width_per_group,
                    "DEPTH": depth
                },
                "RPN": {
                    "HEAD_NAME": rpn_head_name,
                    "IN_FEATURES": rpn_in_features,
                    "BOUNDARY_THRESH": rpn_boundary_thresh,
                    "IOU_THRESHOLDS": rpn_iou_thresholds,
                    "IOU_LABELS": rpn_iou_labels,
                    "BATCH_SIZE_PER_IMAGE": rpn_batch_size_per_image,
                    "POSITIVE_FRACTION": rpn_positive_fraction,
                    "BBOX_REG_LOSS_TYPE": rpn_bbox_reg_loss_type,
                    "BBOX_REG_LOSS_WEIGHT": rpn_bbox_reg_loss_weight,
                    "BBOX_REG_WEIGHTS": rpn_bbox_reg_weights,
                    "SMOOTH_L1_BETA": rpn_smooth_l1_beta,
                    "LOSS_WEIGHT": rpn_loss_weight,
                    "PRE_NMS_TOPK_TRAIN": rpn_pre_nms_topk_train,
                    "PRE_NMS_TOPK_TEST": rpn_pre_nms_topk_test,
                    "POST_NMS_TOPK_TRAIN": rpn_post_nms_topk_train,
                    "POST_NMS_TOPK_TEST": rpn_post_nms_topk_test,
                    "NMS_THRESH": rpn_nms_thresh,
                    "BBOX_LOSS_WEIGHT": rpn_bbox_loss_weight,
                },

                "ROI_HEADS": {
                    "NAME": roi_head_name,
                    "NUM_CLASSES": roi_head_num_classes,
                    "IN_FEATURES": roi_head_in_features,
                    "IOU_THRESHOLDS": roi_head_iou_thresholds,
                    "IOU_LABELS": roi_head_iou_labels,
                    "BATCH_SIZE_PER_IMAGE": roi_head_batch_size_per_image,
                    "POSITIVE_FRACTION": roi_head_positive_fraction,
                    "SCORE_THRESH_TEST": roi_head_score_thresh_test,
                    "NMS_THRESH_TEST": roi_head_nms_thresh_test,
                    "PROPOSAL_APPEND_GT": roi_head_proposal_append_gt,
                },

                "ROI_BOX_HEAD": {
                    "NAME": box_head_name,
                    "BBOX_REG_LOSS_TYPE": box_head_bbox_reg_loss_type,
                    "BBOX_REG_LOSS_WEIGHT": box_head_bbox_reg_loss_weight,
                    "BBOX_REG_WEIGHTS": box_head_bbox_reg_weights,
                    "SMOOTH_L1_BETA": box_head_smooth_l1_beta,
                    "POOLER_RESOLUTION": box_head_pooler_resolution,
                    "POOLER_SAMPLING_RATIO": box_head_pooler_sampling_ratio,
                    "POOLER_TYPE": box_head_pooler_type,
                    "NUM_FC": box_head_num_fc,
                    "FC_DIM": box_head_fc_dim,
                    "NUM_CONV": box_head_num_conv,
                    "CONV_DIM": box_head_conv_dim,
                    "NORM": box_head_norm,
                    "CLS_AGNOSTIC_BBOX_REG": box_head_cls_agnostic_bbox_reg,
                    "TRAIN_ON_PRED_BOXES": box_head_train_on_pred_boxes,
                    "BBOX_LOSS_WEIGHT": box_head_bbox_loss_weight,
                },

                "ROI_ATTRIBUTE_HEAD": {  # not in detectron2 default config
                    "OBJ_EMBED_DIM": attribute_head_obj_embed_dim,
                    "FC_DIM": attribute_head_fc_dim,
                    "LOSS_WEIGHT": attribute_head_loss_weight,
                    "NUM_CLASSES": attribute_head_num_classes,
                }
            },
        }

        from allennlp.common.detectron import get_detectron_cfg
        cfg = get_detectron_cfg(None, None, overrides)
        # from pprint import pprint
        # pprint(overrides)
        # pprint(cfg)
        from detectron2.modeling import build_model
        # TODO: Since we use `GeneralizedRCNN` from detectron2, the model initlized 
        # here still has ROI heads and other redunant parameters for RPN.  
        self.model = build_model(cfg)
        from detectron2.checkpoint import DetectionCheckpointer
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()

    def forward(self, images):

        # handle the single-image case
        if not isinstance(images, list):
            return self.__call__([images])[0]

        images = [self._to_model_input(i) for i in images]
        images = self.model.preprocess_image(images)

        features = self.model.backbone(images.tensor)

        # RPN
        proposals, _ = self.model.proposal_generator(images, features, None)


        # pooled features and box predictions
        _, pooled_features, pooled_features_fc6 = self.model.roi_heads.get_roi_features(
            features, proposals
        )
        predictions = self.model.roi_heads.box_predictor(pooled_features)

        # TODO: current implementation below assumes batch_size=1. Make it handle arbitrary batch size inputs
        cls_probs = F.softmax(predictions[0], dim=-1)
        cls_probs = cls_probs[:, :-1]  # background is last
        predictions, r_indices = self.model.roi_heads.box_predictor.inference(
            predictions, proposals
        )
        # Create Boxes objects from proposals. Since features are extrracted from
        # the proposal boxes we use them instead of predicted boxes.
        box_type = type(proposals[0].proposal_boxes)
        proposal_bboxes = box_type.cat([p.proposal_boxes for p in proposals])
        proposal_bboxes.tensor = proposal_bboxes.tensor[r_indices]
        predictions[0].set("proposal_boxes", proposal_bboxes)
        predictions[0].remove("pred_boxes")

        # postprocess
        height = inputs[0].get("height")
        width = inputs[0].get("width")
        r = postprocessing.detector_postprocess(predictions[0], height, width)

        bboxes = r.get("proposal_boxes").tensor
        classes = r.get("pred_classes")
        cls_probs = cls_probs[r_indices]
        feature_name == 'fc7'
        if feature_name == "fc6" and pooled_features_fc6 is not None:
            pooled_features = pooled_features_fc6[r_indices]
        else:
            pooled_features = pooled_features[r_indices]

        assert (
            bboxes.size(0)
            == classes.size(0)
            == cls_probs.size(0)
            == pooled_features.size(0)
        )

        # save info and features
        info = {
            "bbox": bboxes.cpu().numpy(),
            "num_boxes": bboxes.size(0),
            "objects": classes.cpu().numpy(),
            "image_height": r.image_size[0],
            "image_width": r.image_size[1],
            "cls_prob": cls_probs.cpu().numpy(),
            "features": pooled_features.cpu().numpy()
        }

        return torch.zeros(images.size(0), 0, 4, dtype=torch.float32, device=images.device)


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
