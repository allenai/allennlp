from typing import Dict, Any, Optional

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model

from detectron2.config import CfgNode

def apply_dict_to_cfg(d: Dict[str, Any], c: CfgNode) -> None:
    for key, value in d.items():
        key = key.upper()
        if isinstance(value, dict):
            apply_dict_to_cfg(value, c[key])
        else:
            c[key] = value

@Model.register("detectron")
class Detectron(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        train: Optional[bool] = True,
        builtin_config_file: Optional[str] = None,
        yaml_config_file: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        cfg = get_cfg()
        from detectron2.model_zoo import get_config_file
        if builtin_config_file is not None:
            cfg.merge_from_file(get_config_file(builtin_config_file))
        if yaml_config_file is not None:
            cfg.merge_from_file(yaml_config_file)
        if train:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(builtin_config_file)
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
        if overrides is not None:
            apply_dict_to_cfg(overrides, cfg)
        cfg.freeze()
        
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer

        self.detectron_model = build_model(cfg)
        DetectionCheckpointer(self.detectron_model).load(cfg.MODEL.WEIGHTS)

        from detectron2.utils.events import EventStorage
        self.detectron_event_storage = EventStorage()

    def forward(  # type: ignore
        self, images: Any, proposals: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        self.detectron_event_storage.step()
        self.detectron_model.train()

        with self.detectron_event_storage:
            images = self.detectron_model.preprocess_image(images)
            features = self.detectron_model.backbone(images.tensor)
            
            if proposals is None:
                proposals, _ = self.detectron_model.proposal_generator(images, features, None)

            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.detectron_model.roi_heads.in_features]

        return {"loss": torch.tensor(0.0)}
