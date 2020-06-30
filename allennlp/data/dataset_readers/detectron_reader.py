from overrides import overrides
from typing import Dict, Optional

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance


@DatasetReader.register("detectron")
class DetectronDatasetReader(DatasetReader):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from detectron2.data import DatasetMapper
        from detectron2.config import get_cfg
        self.dataset_mapper = DatasetMapper(get_cfg())

    @overrides
    def _read(self, file_path):
        from detectron2.data.datasets import load_coco_json
        instances = load_coco_json(file_path, "/Users/dirkg/Documents/detectron_datasets/coco_tiny/images")
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
