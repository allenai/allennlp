from typing import Dict, Any

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model


@Model.register("detectron")
class Detectron(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        from detectron2.model_zoo import model_zoo
        self.detectron_model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        from detectron2.utils.events import EventStorage
        self.detectron_event_storage = EventStorage()

    def forward(  # type: ignore
        self, image: Any
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            self.detectron_event_storage.step()
            self.detectron_model.eval()
            with self.detectron_event_storage:
                r = self.detectron_model.inference(image)
        return {"loss": torch.tensor(0.0)}
