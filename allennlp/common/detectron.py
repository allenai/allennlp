from typing import Any, Dict, Optional, NamedTuple, List, Tuple

import torch
from torch import nn
from detectron2.config import CfgNode
from detectron2.data import DatasetMapper
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    build_box_head,
    build_mask_head,
    select_foreground_proposals,
    ROI_HEADS_REGISTRY,
    ROIHeads,
    Res5ROIHeads,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.poolers import ROIPooler


class DetectronConfig(NamedTuple):
    builtin_config_file: Optional[str] = None
    yaml_config_file: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None


class DetectronPipeline(NamedTuple):
    cfg: CfgNode
    mapper: DatasetMapper
    model: nn.Module


def update(d, u):
    for k, v in u.items():
        from collections.abc import Mapping

        if isinstance(v, Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def convert_to_dict(cfg_node, key_list: Optional[List[str]] = None):
    if key_list is None:
        key_list = []
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


def get_cfg(
    builtin_config_file: Optional[str] = None,
    yaml_config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    freeze: bool = True,
) -> CfgNode:
    from detectron2.config import get_cfg

    cfg = get_cfg()
    from detectron2.model_zoo import get_config_file

    if builtin_config_file is not None:
        cfg.merge_from_file(get_config_file(builtin_config_file))
    if yaml_config_file is not None:
        cfg.merge_from_file(yaml_config_file)
    if overrides is not None:
        old_dict = convert_to_dict(cfg, [])
        new_dict = update(old_dict, overrides)
        cfg = CfgNode(new_dict)

    import torch

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    if freeze:
        cfg.freeze()
    return cfg


def load_checkpoint(checkpoint_name: str):
    """
    Download and cache a tarfile containing detectron checkpoint (_weights.th)
    It might be better to replace this with native utility functions like
    load_archive (https://github.com/allenai/allennlp/blob/master/allennlp/models/archival.py#L132)
    later, where the tarfile includes 1) config.json 2) _weights.th
    """
    gs_url = "https://storage.googleapis.com/allennlp-public-models"
    detectron_checkpoint_url = f"{gs_url}/{checkpoint_name}.tar.gz!{checkpoint_name}/_weights.th"
    from allennlp.common.file_utils import cached_path

    return cached_path(detectron_checkpoint_url, extract_archive=True)


_pipeline_cache: Dict[DetectronConfig, DetectronPipeline] = {}


def get_pipeline(
    builtin_config_file: Optional[str] = None,
    yaml_config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    make_copy: bool = True,
) -> DetectronPipeline:
    global _pipeline_cache

    cfg = get_cfg(builtin_config_file, yaml_config_file, overrides)
    spec = cfg.dump()  # Easiest way to get a hashable snapshot of CfgNode.
    pipeline = _pipeline_cache.get(spec, None)
    if pipeline is None:
        from detectron2.data import DatasetMapper

        mapper = DatasetMapper(cfg)
        from detectron2.modeling import build_model

        model = build_model(cfg)
        from detectron2.checkpoint import DetectionCheckpointer

        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval()
        pipeline = DetectronPipeline(cfg, mapper, model)
        _pipeline_cache[spec] = pipeline

    if make_copy:
        import copy

        return copy.deepcopy(pipeline)
    else:
        return pipeline


class DetectronFlatParameters(NamedTuple):
    meta_architecture: str = "GeneralizedRCNN"
    device: str = "cpu"
    weights: str = "RCNN-X152-C4-2020-07-18"

    attribute_on: bool = True  # not in detectron2 default config
    max_attr_per_ins: int = 16  # not in detectron2 default config

    pixel_mean: List[float] = [103.530, 116.280, 123.675]
    pixel_std: List[float] = [1.0, 1.0, 1.0]

    # Resnet (ResNeXt) backbone
    stride_in_1x1: bool = False  # different from default (True)
    num_groups: int = 32  # different from default (1)
    width_per_group: int = 8  # different from default (64)
    depth: int = 152  # different from default (50)

    # RPN
    rpn_head_name: str = "StandardRPNHead"
    rpn_in_features: List[str] = ["res4"]
    rpn_boundary_thresh: int = 0
    rpn_iou_thresholds: List[float] = [0.3, 0.7]
    rpn_iou_labels: List[int] = [0, -1, 1]
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    rpn_bbox_reg_loss_type: str = "smooth_l1"
    rpn_bbox_reg_loss_weight: float = 1.0  # different from default (-1)
    rpn_bbox_reg_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    rpn_smooth_l1_beta: float = 0.1111  # different from default (0.0)
    rpn_loss_weight: float = 1.0
    rpn_pre_nms_topk_train: int = 12000
    rpn_pre_nms_topk_test: int = 6000
    rpn_post_nms_topk_train: int = 2000
    rpn_post_nms_topk_test: int = 1000
    rpn_nms_thresh: float = 0.7
    rpn_bbox_loss_weight: float = 1.0  # not in detectron2 default config

    # ROI head
    roi_head_name: str = "AttributeRes5ROIHeads"  # different from default (Res5ROIHeads)
    roi_head_num_classes: int = 1600  # different from default (80)
    roi_head_in_features: List[str] = ["res4"]
    roi_head_iou_thresholds: List[float] = [0.5]
    roi_head_iou_labels: List[int] = [0, 1]
    roi_head_batch_size_per_image: int = 512
    roi_head_positive_fraction: float = 0.25
    roi_head_score_thresh_test: float = 0.05
    roi_head_nms_thresh_test: float = 0.5
    roi_head_proposal_append_gt: bool = True

    # ROI Box head
    box_head_name: str = "FastRCNNConvFCHead"  # different from default ("")
    box_head_bbox_reg_loss_type: str = "smooth_l1"
    box_head_bbox_reg_loss_weight: float = 1.0
    box_head_bbox_reg_weights: Tuple[float, ...] = (10.0, 10.0, 5.0, 5.0)
    box_head_smooth_l1_beta: float = 1.0  # different from default (0.0)
    box_head_pooler_resolution: int = 7  # different from default (14)
    box_head_pooler_sampling_ratio: int = 2  # different from default (0)
    box_head_pooler_type: str = "ROIAlignV2"
    box_head_num_fc: int = 2  # different from default (0)
    box_head_fc_dim: int = 1024
    box_head_num_conv: int = 0
    box_head_conv_dim: int = 256
    box_head_norm: str = ""
    box_head_cls_agnostic_bbox_reg: bool = False
    box_head_train_on_pred_boxes: bool = False
    box_head_bbox_loss_weight: float = 1.0  # not in detectron2 default config

    # Box attribute classification head (not in detectron2 default config)
    attribute_head_obj_embed_dim: int = 256
    attribute_head_fc_dim: int = 512
    attribute_head_loss_weight: float = 0.2
    attribute_head_num_classes: int = 400

    # Test
    test_detections_per_image: int = 36  # different from default (100)


def get_pipeline_from_flat_parameters(
    fp: DetectronFlatParameters, make_copy: bool = True
) -> DetectronPipeline:
    if fp.weights is None:
        weights = None
    else:
        weights = load_checkpoint(fp.weights)

    overrides = {
        "VERSION": 2,
        "INPUT": {"MAX_ATTR_PER_INS": fp.max_attr_per_ins},
        "MODEL": {
            "DEVICE": fp.device,
            "WEIGHTS": weights,
            "META_ARCHITECTURE": fp.meta_architecture,
            "ATTRIBUTE_ON": fp.attribute_on,
            "PIXEL_MEAN": fp.pixel_mean,
            "PIXEL_STD": fp.pixel_std,
            "PROPOSAL_GENERATOR": {"NAME": "RPN"},
            "RESNETS": {
                "STRIDE_IN_1X1": fp.stride_in_1x1,
                "NUM_GROUPS": fp.num_groups,
                "WIDTH_PER_GROUP": fp.width_per_group,
                "DEPTH": fp.depth,
            },
            "RPN": {
                "HEAD_NAME": fp.rpn_head_name,
                "IN_FEATURES": fp.rpn_in_features,
                "BOUNDARY_THRESH": fp.rpn_boundary_thresh,
                "IOU_THRESHOLDS": fp.rpn_iou_thresholds,
                "IOU_LABELS": fp.rpn_iou_labels,
                "BATCH_SIZE_PER_IMAGE": fp.rpn_batch_size_per_image,
                "POSITIVE_FRACTION": fp.rpn_positive_fraction,
                "BBOX_REG_LOSS_TYPE": fp.rpn_bbox_reg_loss_type,
                "BBOX_REG_LOSS_WEIGHT": fp.rpn_bbox_reg_loss_weight,
                "BBOX_REG_WEIGHTS": fp.rpn_bbox_reg_weights,
                "SMOOTH_L1_BETA": fp.rpn_smooth_l1_beta,
                "LOSS_WEIGHT": fp.rpn_loss_weight,
                "PRE_NMS_TOPK_TRAIN": fp.rpn_pre_nms_topk_train,
                "PRE_NMS_TOPK_TEST": fp.rpn_pre_nms_topk_test,
                "POST_NMS_TOPK_TRAIN": fp.rpn_post_nms_topk_train,
                "POST_NMS_TOPK_TEST": fp.rpn_post_nms_topk_test,
                "NMS_THRESH": fp.rpn_nms_thresh,
                "BBOX_LOSS_WEIGHT": fp.rpn_bbox_loss_weight,
            },
            "ROI_HEADS": {
                "NAME": fp.roi_head_name,
                "NUM_CLASSES": fp.roi_head_num_classes,
                "IN_FEATURES": fp.roi_head_in_features,
                "IOU_THRESHOLDS": fp.roi_head_iou_thresholds,
                "IOU_LABELS": fp.roi_head_iou_labels,
                "BATCH_SIZE_PER_IMAGE": fp.roi_head_batch_size_per_image,
                "POSITIVE_FRACTION": fp.roi_head_positive_fraction,
                "SCORE_THRESH_TEST": fp.roi_head_score_thresh_test,
                "NMS_THRESH_TEST": fp.roi_head_nms_thresh_test,
                "PROPOSAL_APPEND_GT": fp.roi_head_proposal_append_gt,
            },
            "ROI_BOX_HEAD": {
                "NAME": fp.box_head_name,
                "BBOX_REG_LOSS_TYPE": fp.box_head_bbox_reg_loss_type,
                "BBOX_REG_LOSS_WEIGHT": fp.box_head_bbox_reg_loss_weight,
                "BBOX_REG_WEIGHTS": fp.box_head_bbox_reg_weights,
                "SMOOTH_L1_BETA": fp.box_head_smooth_l1_beta,
                "POOLER_RESOLUTION": fp.box_head_pooler_resolution,
                "POOLER_SAMPLING_RATIO": fp.box_head_pooler_sampling_ratio,
                "POOLER_TYPE": fp.box_head_pooler_type,
                "NUM_FC": fp.box_head_num_fc,
                "FC_DIM": fp.box_head_fc_dim,
                "NUM_CONV": fp.box_head_num_conv,
                "CONV_DIM": fp.box_head_conv_dim,
                "NORM": fp.box_head_norm,
                "CLS_AGNOSTIC_BBOX_REG": fp.box_head_cls_agnostic_bbox_reg,
                "TRAIN_ON_PRED_BOXES": fp.box_head_train_on_pred_boxes,
                "BBOX_LOSS_WEIGHT": fp.box_head_bbox_loss_weight,
            },
            "ROI_ATTRIBUTE_HEAD": {  # not in detectron2 default config
                "OBJ_EMBED_DIM": fp.attribute_head_obj_embed_dim,
                "FC_DIM": fp.attribute_head_fc_dim,
                "LOSS_WEIGHT": fp.attribute_head_loss_weight,
                "NUM_CLASSES": fp.attribute_head_num_classes,
            },
        },
        "TEST": {"DETECTIONS_PER_IMAGE": fp.test_detections_per_image},
    }

    return get_pipeline(None, None, overrides, make_copy)


class AttributePredictor(nn.Module):
    """
    Head for attribute prediction, including feature/score computation and
    loss computation.
    """

    def __init__(self, cfg, input_dim):
        super().__init__()

        self.num_objs = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.obj_embed_dim = cfg.MODEL.ROI_ATTRIBUTE_HEAD.OBJ_EMBED_DIM
        self.fc_dim = cfg.MODEL.ROI_ATTRIBUTE_HEAD.FC_DIM
        self.num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_CLASSES
        self.max_attr_per_ins = cfg.INPUT.MAX_ATTR_PER_INS
        self.loss_weight = cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT

        # object class embedding, including the background class
        self.obj_embed = nn.Embedding(self.num_objs + 1, self.obj_embed_dim)
        input_dim += self.obj_embed_dim
        self.fc = nn.Sequential(nn.Linear(input_dim, self.fc_dim), nn.ReLU())
        self.attr_score = nn.Linear(self.fc_dim, self.num_attributes)
        nn.init.normal_(self.attr_score.weight, std=0.01)
        nn.init.constant_(self.attr_score.bias, 0)

    def forward(self, x, obj_labels):
        attr_feat = torch.cat((x, self.obj_embed(obj_labels)), dim=1)
        return self.attr_score(self.fc(attr_feat))

    def loss(self, score, label):
        n = score.shape[0]
        score = score.unsqueeze(1)
        score = score.expand(n, self.max_attr_per_ins, self.num_attributes).contiguous()
        score = score.view(-1, self.num_attributes)
        inv_weights = (
            (label >= 0).sum(dim=1).repeat(self.max_attr_per_ins, 1).transpose(0, 1).flatten()
        )
        weights = inv_weights.float().reciprocal()
        weights[weights > 1] = 0.0
        n_valid = len((label >= 0).sum(dim=1).nonzero())
        label = label.view(-1)
        attr_loss = F.cross_entropy(score, label, reduction="none", ignore_index=-1)
        attr_loss = (attr_loss * weights).view(n, -1).sum(dim=1)

        if n_valid > 0:
            attr_loss = attr_loss.sum() * self.loss_weight / n_valid
        else:
            attr_loss = attr_loss.sum() * 0.0
        return {"loss_attr": attr_loss}


class AttributeROIHeads(ROIHeads):
    """
    An extension of ROIHeads to include attribute prediction.
    """

    def forward_attribute_loss(self, proposals, box_features):
        proposals, fg_selection_attributes = select_foreground_proposals(
            proposals, self.num_classes
        )
        attribute_features = box_features[torch.cat(fg_selection_attributes, dim=0)]
        obj_labels = torch.cat([p.gt_classes for p in proposals])
        attribute_labels = torch.cat([p.gt_attributes for p in proposals], dim=0)
        attribute_scores = self.attribute_predictor(attribute_features, obj_labels)
        return self.attribute_predictor.loss(attribute_scores, attribute_labels)


@ROI_HEADS_REGISTRY.register()
class AttributeRes5ROIHeads(AttributeROIHeads, Res5ROIHeads):
    """
    An extension of Res5ROIHeads to include attribute prediction.
    This code was modified based on the repo:
    https://github.com/vedanuj/grid-feats-vqa/blob/master/grid_feats/roi_heads.py
    """

    def __init__(self, cfg, input_shape):
        super(Res5ROIHeads, self).__init__(cfg)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        assert len(self.in_features) == 1
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on = cfg.MODEL.MASK_ON
        self.attribute_on = cfg.MODEL.ATTRIBUTE_ON
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

        if self.attribute_on:
            self.attribute_predictor = AttributePredictor(cfg, out_channels)

    def forward(self, images, features, proposals, targets=None):
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])
        predictions = self.box_predictor(feature_pooled)

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            if self.attribute_on:
                losses.update(self.forward_attribute_loss(proposals, feature_pooled))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def get_conv5_features(self, features):
        features = [features[f] for f in self.in_features]
        return self.res5(features[0])

    def get_roi_features(self, features, proposals):
        assert len(self.in_features) == 1

        features = [features[f] for f in self.in_features]
        box_features = self._shared_roi_transform(features, [x.proposal_boxes for x in proposals])
        pooled_features = box_features.mean(dim=[2, 3])
        return box_features, pooled_features


@ROI_HEADS_REGISTRY.register()
class AttributeStandardROIHeads(AttributeROIHeads, StandardROIHeads):
    """
    An extension of StandardROIHeads to include attribute prediction.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        self.attribute_on = cfg.MODEL.ATTRIBUTE_ON

        in_channels = [input_shape[f].channels for f in self.in_features]
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(cfg, self.box_head.output_shape)

        if self.attribute_on:
            self.attribute_predictor = AttributePredictor(cfg, self.box_head.output_shape.channels)

    def _forward_box(self, features, proposals):
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses = self.box_predictor.losses(predictions, proposals)
            if self.attribute_on:
                losses.update(self.forward_attribute_loss(proposals, box_features))
                del box_features

            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def get_conv5_features(self, features):
        assert len(self.in_features) == 1

        features = [features[f] for f in self.in_features]
        return features[0]
