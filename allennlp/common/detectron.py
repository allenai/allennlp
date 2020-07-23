from typing import Any, Dict, Optional, NamedTuple

from detectron2.config import CfgNode


class DetectronConfig(NamedTuple):
    builtin_config_file: Optional[str] = None
    yaml_config_file: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None


# def apply_dict_to_cfg(d: Dict[str, Any], c: CfgNode) -> None:
#     for key, value in d.items():
#         key = key.upper()
#         if isinstance(value, dict):
#             try:
#                 apply_dict_to_cfg(value, c[key])
#             except KeyError:
#                 c[key] = {}
#                 apply_dict_to_cfg(value, c[key])
#         else:
#             c[key] = value

import collections.abc
def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def convert_to_dict(cfg_node, key_list = []):
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


def get_detectron_cfg(
    builtin_config_file: Optional[str] = None,
    yaml_config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    freeze: bool = True,
) -> CfgNode:
    from detectron2.config import get_cfg

    cfg = get_cfg()
    # add_attribute_config(cfg)  # TODO: What is this? It's not general, not configurable.
    from detectron2.model_zoo import get_config_file

    if builtin_config_file is not None:
        cfg.merge_from_file(get_config_file(builtin_config_file))
    if yaml_config_file is not None:
        cfg.merge_from_file(yaml_config_file)
    if overrides is not None:
        # print('old_dict')
        # from pprint import pprint
        old_dict = convert_to_dict(cfg, [])
        # old_string = cfg.dump()
        # pprint(old_dict)
        new_dict = update(old_dict, overrides)
        # print('new_dict')
        # pprint(new_dict)
        cfg = CfgNode(new_dict)

        # apply_dict_to_cfg(overrides, cfg)

    import torch
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    if freeze:
        cfg.freeze()
    return cfg


# def add_attribute_config(cfg):
#     """
#     Add config for attribute prediction.
#     """
#     # What feature type we want, region or grid:
#     cfg.MODEL.FEATURE_TYPE = "region"
#     # Whether to have attribute prediction
#     cfg.MODEL.ATTRIBUTE_ON = False
#     # Maximum number of attributes per foreground instance
#     cfg.INPUT.MAX_ATTR_PER_INS = 16
#     # ------------------------------------------------------------------------ #
#     # Attribute Head
#     # -----------------------------------------------------------------------  #
#     cfg.MODEL.ROI_ATTRIBUTE_HEAD = CfgNode()
#     # Dimension for object class embedding, used in conjunction with
#     # visual features to predict attributes
#     cfg.MODEL.ROI_ATTRIBUTE_HEAD.OBJ_EMBED_DIM = 256
#     # Dimension of the hidden fc layer of the input visual features
#     cfg.MODEL.ROI_ATTRIBUTE_HEAD.FC_DIM = 512
#     # Loss weight for attribute prediction, 0.2 is best per analysis
#     cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT = 0.2
#     # Number of classes for attributes
#     cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_CLASSES = 400

#     """
#     Add config for box regression loss adjustment.
#     """
#     # Loss weights for RPN box regression
#     cfg.MODEL.RPN.BBOX_LOSS_WEIGHT = 1.0
#     # Loss weights for R-CNN box regression
#     cfg.MODEL.ROI_BOX_HEAD.BBOX_LOSS_WEIGHT = 1.0
