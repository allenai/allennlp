import logging
from typing import Dict, Optional

from overrides import overrides
import numpy as np
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import (
    TextEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
)
from allennlp.training.metrics import CategoricalAccuracy


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
    text_embeddings : `TextEmbeddings`
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
        text_embeddings: TextEmbeddings,
        image_embeddings: ImageFeatureEmbeddings,
        encoder: BiModalEncoder,
        pooled_output_dim: int,
        fusion_method: str = "sum",
        dropout: float = 0.1,
        label_namespace: str = "labels",
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
        )

        self.accuracy = CategoricalAccuracy()

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        hypothesis: TextFieldTensors,
        label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        return super().forward(
            box_features,
            box_coordinates,
            box_mask,
            text=hypothesis,
            label=label,
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
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.accuracy.get_metric(reset)
        return {"accuracy": result}

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
