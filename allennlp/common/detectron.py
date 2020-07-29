from typing import Any, Dict, Optional, NamedTuple, List, Tuple
from torch import nn
from detectron2.config import CfgNode
from detectron2.data import DatasetMapper


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
    detectron_checkpoint_url = f'{gs_url}/{checkpoint_name}.tar.gz!{checkpoint_name}/_weights.th'
    from allennlp.common.file_utils import cached_path
    return cached_path(detectron_checkpoint_url, extract_archive=True)


_pipeline_cache: Dict[DetectronConfig, DetectronPipeline] = {}


def get_pipeline(
    builtin_config_file: Optional[str] = None,
    yaml_config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    make_copy: bool = True
) -> DetectronPipeline:
    global _pipeline_cache

    cfg = get_cfg(builtin_config_file, yaml_config_file, overrides)
    spec = cfg.dump()   # Easiest way to get a hashable snapshot of CfgNode.
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
    rpn_bbox_reg_loss_type: str = 'smooth_l1'
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
    test_detections_per_image: int = 36 # different from default (100)


def get_pipeline_from_flat_parameters(
    fp: DetectronFlatParameters,
    make_copy: bool = True
) -> DetectronPipeline:
    if fp.weights is None:
        weights = None
    else:
        weights = load_checkpoint(fp.weights)

    overrides = {
        "VERSION": 2,
        "INPUT": {
            "MAX_ATTR_PER_INS": fp.max_attr_per_ins,
        },
        "MODEL": {
            "DEVICE": fp.device,
            "WEIGHTS": weights,
            "META_ARCHITECTURE": fp.meta_architecture,
            "ATTRIBUTE_ON": fp.attribute_on,

            "PIXEL_MEAN": fp.pixel_mean,
            "PIXEL_STD": fp.pixel_std,

            "PROPOSAL_GENERATOR": {
                "NAME": "RPN",
            },
            "RESNETS": {
                "STRIDE_IN_1X1": fp.stride_in_1x1,
                "NUM_GROUPS": fp.num_groups,
                "WIDTH_PER_GROUP": fp.width_per_group,
                "DEPTH": fp.depth
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

            "TEST": {
                "DETECTIONS_PER_IMAGE": fp.test_detections_per_image,
            }
        },
    }

    return get_pipeline(None, None, overrides, make_copy)
