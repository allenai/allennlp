from typing import Dict, Optional

import torch
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.heads.head import Head


@Head.register("visual_entailment")
class VisualEntailmentHead(Head):
    def __init__(self, vocab: Vocabulary, embedding_dim: int, label_namespace: str = "labels"):
        super().__init__(vocab)

        num_labels = vocab.get_vocab_size(label_namespace)
        self.label_namespace = label_namespace
        self.classifier = torch.nn.Linear(embedding_dim, num_labels)

        from allennlp.training.metrics import CategoricalAccuracy
        from allennlp.training.metrics import FBetaMeasure

        self.accuracy = CategoricalAccuracy()
        self.fbeta = FBetaMeasure(beta=1.0, average="macro")

    @overrides
    def forward(
        self,  # type: ignore
        encoded_boxes: torch.Tensor,
        encoded_boxes_mask: torch.Tensor,
        encoded_boxes_pooled: torch.Tensor,
        encoded_text: torch.Tensor,
        encoded_text_mask: torch.Tensor,
        encoded_text_pooled: torch.Tensor,
        pooled_boxes_and_text: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.classifier(pooled_boxes_and_text)
        probs = torch.softmax(logits, dim=-1)

        output = {"logits": logits, "probs": probs}

        assert label_weights is None
        if labels is not None:
            output["loss"] = torch.nn.functional.cross_entropy(logits, labels) / logits.size(0)
            self.accuracy(logits, labels)
            self.fbeta(probs, labels)

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.fbeta.get_metric(reset)
        result["acc"] = self.accuracy.get_metric(reset)
        return result
