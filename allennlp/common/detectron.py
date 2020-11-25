from collections.abc import Mapping
from typing import Any, Dict, Optional, List, Union

from detectron2 import config, model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper
from detectron2.modeling import build_model
from detectron2.modeling.backbone import Backbone
from detectron2.structures.image_list import ImageList
from torch import nn
import torch

from allennlp.common.registrable import Registrable
from allennlp.common.file_utils import cached_path


class DetectronConfig(Registrable):
    """
    This is an immutable wrapper around a Detectron2 `CfgNode`.
    """

    default_implementation = "from_model_zoo"

    def __init__(
        self, cfg: config.CfgNode, overrides: Dict[str, Any] = None, device: str = None
    ) -> None:
        self.cfg = cfg

        if device is not None:
            self.cfg.MODEL.DEVICE = device
        elif not torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = "cpu"

        if overrides is not None:
            self.update(overrides)

        self.cfg.freeze()

    @property
    def device(self) -> str:
        return self.cfg.MODEL.DEVICE

    def __getattr__(self, key: str):
        if key.isupper():
            return getattr(self.cfg, key)
        raise AttributeError

    def update(self, other: Union["DetectronConfig", Dict[str, Any]]) -> None:
        def _update(d, u):
            for k, v in u.items():
                if isinstance(v, Mapping):
                    d[k] = _update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        other_as_dict = other if isinstance(other, dict) else other.as_dict()
        new_dict = _update(self.as_dict(), other_as_dict)
        self.cfg = config.CfgNode(new_dict)

    def as_dict(self) -> Dict[str, Any]:
        def _convert_to_dict(cfg_node, key_list: Optional[List[str]] = None):
            if key_list is None:
                key_list = []
            if not isinstance(cfg_node, config.CfgNode):
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = _convert_to_dict(v, key_list + [k])
                return cfg_dict

        return _convert_to_dict(self.cfg)

    def build_dataset_mapper(self) -> DatasetMapper:
        """
        Build a detectron `DatasetMapper` from this config.
        """
        return DatasetMapper(self.cfg)

    def build_backbone(self) -> Backbone:
        """
        Build a detectron backbone from this config.
        """
        return self.build_model().backbone

    def build_model(self) -> nn.Module:
        """
        Build a detectron model from this config.
        """
        model = build_model(self.cfg)
        if self.cfg.MODEL.WEIGHTS is not None:
            DetectionCheckpointer(model).load(self.cfg.MODEL.WEIGHTS)
        model.eval()
        return model

    @classmethod
    def from_model_zoo(
        cls,
        config_path: str,
        overrides: Dict[str, Any] = None,
        device: str = None,
    ) -> "DetectronConfig":
        """
        Instantiate a config from a model in Detectron's [model zoo]
        (https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md).

        `config_path` should be a path to a YAML config in detectron2's [config]
        (https://github.com/facebookresearch/detectron2/tree/master/configs) directory,
        relative to the directory.

        For example, `config_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"`.
        """
        cfg_file = model_zoo.get_config_file(config_path)
        cfg = config.get_cfg()
        cfg.merge_from_file(cfg_file)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
        return cls(cfg, overrides=overrides, device=device)

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any], device: str = None) -> "DetectronConfig":
        """
        Instantiate a config from a raw config dictionary.
        """
        return cls(config.get_cfg(), overrides=cfg_dict, device=device)


DetectronConfig.register("from_model_zoo", constructor="from_model_zoo")(DetectronConfig)
DetectronConfig.register("from_dict", constructor="from_dict")(DetectronConfig)
DetectronConfig.register("from_checkpoint", constructor="from_checkpoint")(DetectronConfig)
DetectronConfig.register("from_flat_parameters", constructor="from_flat_parameters")(
    DetectronConfig
)


def pack_images(
    images: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
) -> torch.Tensor:
    """
    Pack a list of images into a single padded tensor.

    This is just a wrapper around `detectron2.structures.ImageList`.
    """
    return ImageList.from_tensors(
        images, size_divisibility=size_divisibility, pad_value=pad_value
    ).tensor
