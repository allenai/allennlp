import logging
from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import (
    TransformerEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
)
from allennlp.nn import util

from allennlp.models.vision_text_model import VisionTextModel


logger = logging.getLogger(__name__)


@Model.register("vqa_vilbert")
@Model.register("vqa_vilbert_from_huggingface", constructor="from_huggingface_model_name")
class VqaVilbert(VisionTextModel):
    """
    Model for VQA task based on the VilBERT paper.

    # Parameters

    vocab : `Vocabulary`
    text_embeddings : `TransformerEmbeddings`
    image_embeddings : `ImageFeatureEmbeddings`
    encoder : `BiModalEncoder`
    pooled_output_dim : `int`
    fusion_method : `str`, optional (default = `"sum"`)
    dropout : `float`, optional (default = `0.1`)
    label_namespace : `str`, optional (default = `answers`)
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_embeddings: TransformerEmbeddings,
        image_embeddings: ImageFeatureEmbeddings,
        encoder: BiModalEncoder,
        pooled_output_dim: int,
        fusion_method: str = "sum",
        dropout: float = 0.1,
        label_namespace: str = "answers",
        *,
        ignore_text: bool = False,
        ignore_image: bool = False
    ) -> None:
        super().__init__(
            vocab,
            text_embeddings,
            image_embeddings,
            encoder,
            pooled_output_dim,
            fusion_method,
            dropout,
            label_namespace,
            is_multilabel=True,
            ignore_text=ignore_text,
            ignore_image=ignore_image,
        )

        from allennlp.training.metrics import F1MultiLabelMeasure
        from allennlp.training.metrics.vqa import VqaMeasure

        self.f1_metric = F1MultiLabelMeasure(average="micro")
        self.vqa_metric = VqaMeasure()

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        question: TextFieldTensors,
        labels: Optional[torch.Tensor] = None,
        label_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        return super().forward(
            box_features,
            box_coordinates,
            box_mask,
            text=question,
            labels=labels,
            label_weights=label_weights,
        )

    @overrides
    def _compute_loss_and_metrics(
        self,
        batch_size: int,
        outputs: torch.Tensor,
        label: torch.Tensor,
        label_weights: Optional[torch.Tensor] = None,
    ):
        if label is not None and label_weights is not None:
            logits = outputs["logits"]
            label_mask = label > 1  # 0 is padding, 1 is OOV, which we want to ignore

            weighted_labels = util.masked_index_replace(
                logits.new_zeros(logits.size() + (1,)),
                label.clamp(min=0),
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

            outputs["loss"] = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, weighted_labels, weight=binary_label_mask, reduction="sum"
                )
                / batch_size
            )

            self.f1_metric(logits, weighted_labels, binary_label_mask.bool())
            self.vqa_metric(logits, label, label_weights)

        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.f1_metric.get_metric(reset)
        result["vqa_score"] = self.vqa_metric.get_metric(reset)["score"]
        return result

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        batch_tokens = []
        for batch_index, batch in enumerate(output_dict["probs"]):
            tokens = {}
            for i, prob in enumerate(batch):
                tokens[self.vocab.get_token_from_index(i, self.label_namespace)] = float(prob)
            batch_tokens.append(tokens)
        output_dict["tokens"] = batch_tokens
        return output_dict

    default_predictor = "vilbert_vqa"
