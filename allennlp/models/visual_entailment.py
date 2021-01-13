import logging
from typing import Dict, Optional

from overrides import overrides
import numpy as np
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import (
    TransformerEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
)
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import FBetaMeasure


from allennlp.models.vision_text_model import VisionTextModel

logger = logging.getLogger(__name__)


@Model.register("ve_vilbert")
@Model.register("ve_vilbert_from_huggingface", constructor="from_huggingface_model_name")
class VisualEntailmentModel(VisionTextModel):
    """
    Model for visual entailment task based on the paper
    [Visual Entailment: A Novel Task for Fine-Grained Image Understanding]
    (https://api.semanticscholar.org/CorpusID:58981654).

    # Parameters

    vocab : `Vocabulary`
    text_embeddings : `TransformerEmbeddings`
    image_embeddings : `ImageFeatureEmbeddings`
    encoder : `BiModalEncoder`
    pooled_output_dim : `int`
    fusion_method : `str`, optional (default = `"sum"`)
    dropout : `float`, optional (default = `0.1`)
    label_namespace : `str`, optional (default = `labels`)
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
        label_namespace: str = "labels",
        *,
        ignore_text: bool = False,
        ignore_image: bool = False,
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
            is_multilabel=False,
        )

        self.accuracy = CategoricalAccuracy()
        self.fbeta = FBetaMeasure(beta=1.0, average="macro")

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        hypothesis: TextFieldTensors,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        return super().forward(
            box_features,
            box_coordinates,
            box_mask,
            text=hypothesis,
            labels=labels,
            label_weights=None,
        )

    @overrides
    def _compute_loss_and_metrics(
        self,
        batch_size: int,
        outputs: torch.Tensor,
        label: torch.Tensor,
        label_weights: Optional[torch.Tensor] = None,
    ):
        assert label_weights is None
        if label is not None:
            outputs["loss"] = (
                torch.nn.functional.cross_entropy(outputs["logits"], label) / batch_size
            )
            self.accuracy(outputs["logits"], label)
            self.fbeta(outputs["probs"], label)
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.fbeta.get_metric(reset)
        accuracy = self.accuracy.get_metric(reset)
        metrics.update({"accuracy": accuracy})
        return metrics

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        batch_labels = []
        for batch_index, batch in enumerate(output_dict["probs"]):
            labels = np.argmax(batch, axis=-1)
            batch_labels.append(labels)
        output_dict["labels"] = batch_labels
        return output_dict

    default_predictor = "vilbert_ve"
