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
        from torch.nn import Embedding
        self.embedding = Embedding(10, 10)  # dummy

    def forward(  # type: ignore
        self, image: Any
    ) -> Dict[str, torch.Tensor]:
        return {"loss": torch.tensor(0.0)}
