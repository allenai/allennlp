from typing import Any, Dict, Optional, NamedTuple

from detectron2.config import CfgNode


class DetectronConfig(NamedTuple):
    builtin_config_file: Optional[str] = None
    yaml_config_file: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None


def apply_dict_to_cfg(d: Dict[str, Any], c: CfgNode) -> None:
    for key, value in d.items():
        key = key.upper()
        if isinstance(value, dict):
            apply_dict_to_cfg(value, c[key])
        else:
            c[key] = value


def get_detectron_cfg(
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
        apply_dict_to_cfg(overrides, cfg)
    import torch

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    if freeze:
        cfg.freeze()
    return cfg
