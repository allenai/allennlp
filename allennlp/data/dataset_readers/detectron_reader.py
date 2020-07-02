from detectron2.config import CfgNode
from overrides import overrides
from typing import Dict, Optional, Any

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance


def apply_dict_to_cfg(d: Dict[str, Any], c: CfgNode) -> None:
    for key, value in d.items():
        key = key.upper()
        if isinstance(value, dict):
            apply_dict_to_cfg(value, c[key])
        else:
            c[key] = value


@DatasetReader.register("detectron")
class DetectronDatasetReader(DatasetReader):
    def __init__(
        self,
        *,
        builtin_config_file: Optional[str] = None,
        yaml_config_file: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from detectron2.config import get_cfg
        cfg = get_cfg()
        from detectron2.model_zoo import get_config_file
        if builtin_config_file is not None:
            cfg.merge_from_file(get_config_file(builtin_config_file))
        if yaml_config_file is not None:
            cfg.merge_from_file(yaml_config_file)
        if overrides is not None:
            apply_dict_to_cfg(overrides, cfg)
        cfg.freeze()
        from detectron2.data import DatasetMapper
        self.dataset_mapper = DatasetMapper(cfg)

    @overrides
    def _read(self, file_path):
        from detectron2.data.datasets import load_coco_json
        instances = load_coco_json(file_path, "/Users/dirkg/Documents/data/vision/coco_tiny/images")
        for instance in instances:
            instance = self.text_to_instance(instance)
            if instance is not None:
                yield instance

    def text_to_instance(  # type: ignore
        self, detectron_dict: Dict
    ) -> Optional[Instance]:
        from detectron2.data.detection_utils import SizeMismatchError
        from PIL import UnidentifiedImageError
        try:
            model_input = self.dataset_mapper(detectron_dict)
        except (UnidentifiedImageError, SizeMismatchError):
            return None
        from allennlp.data.fields.detectron_field import DetectronField
        return Instance({
            "image": DetectronField(model_input)
        })
