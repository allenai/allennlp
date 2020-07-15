from overrides import overrides
from typing import Dict, Optional, Any

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance


@DatasetReader.register("detectron")
class DetectronDatasetReader(DatasetReader):
    def __init__(
        self,
        *,
        builtin_config_file: Optional[str] = None,
        yaml_config_file: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        image_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from allennlp.common.detectron import get_detectron_cfg

        cfg = get_detectron_cfg(builtin_config_file, yaml_config_file, overrides)
        from detectron2.data import DatasetMapper

        self.dataset_mapper = DatasetMapper(cfg)
        self.image_root = image_root

    @overrides
    def _read(self, file_path):
        from detectron2.data.datasets import load_coco_json

        instances = load_coco_json(file_path, self.image_root)
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

        return Instance({"images": DetectronField(model_input)})
