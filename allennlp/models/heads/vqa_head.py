from typing import Dict, Optional

import torch
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.heads.head import Head


@Head.register("vqa")
class VqaHead(Head):
    def __init__(self, vocab: Vocabulary, embedding_dim: int, label_namespace: str = "answers"):
        super().__init__(vocab)

        num_labels = vocab.get_vocab_size(label_namespace)
        self.label_namespace = label_namespace
        self.classifier = torch.nn.Linear(embedding_dim, num_labels)

        from allennlp.training.metrics import F1MultiLabelMeasure
        from allennlp.training.metrics.vqa import VqaMeasure

        self.f1_metric = F1MultiLabelMeasure(average="micro")
        self.vqa_metric = VqaMeasure()

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

        output = {
            "logits": logits,
            "probs": torch.sigmoid(logits),
        }

        if labels is not None and label_weights is not None:
            label_mask = labels > 1  # 0 is padding, 1 is OOV, which we want to ignore

            from allennlp.nn import util

            weighted_labels = util.masked_index_replace(
                logits.new_zeros(logits.size() + (1,)),
                labels.clamp(min=0),
                label_mask,
                label_weights.unsqueeze(-1),
            ).squeeze(-1)

            # weighted_labels now has shape (batch_size, num_labels).  We need to ignore the first
            # two columns of this in our loss function and accuracy metric.  The first column is a
            # padding label, and the second column is an OOV label.  We want the loss function to
            # be computed on every other label.
            binary_label_mask = weighted_labels.new_ones(logits.size())
            binary_label_mask[:, 0] = 0
            binary_label_mask[:, 1] = 0

            output["loss"] = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, weighted_labels, weight=binary_label_mask, reduction="sum"
            ) / logits.size(0)

            self.f1_metric(logits, weighted_labels, binary_label_mask.bool())
            self.vqa_metric(logits, labels, label_weights)

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.f1_metric.get_metric(reset)
        result["vqa"] = self.vqa_metric.get_metric(reset)["score"]
        return result
