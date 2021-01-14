import logging
from copy import deepcopy
from typing import Dict, List, Optional

from overrides import overrides
import numpy as np
import torch
from transformers import AutoModel

from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import (
    TransformerEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
)

logger = logging.getLogger(__name__)


@Model.register("vision_model")
class VisionTextModel(Model):
    """
    `VisionTextModel` takes as input a single text input and a single image input
    to produce some output. Example tasks include visual question-answering, visual
    entailment, etc.

    # Parameters

    vocab : `Vocabulary`
    text_embeddings : `TransformerEmbeddings`
    image_embeddings : `ImageFeatureEmbeddings`
    encoder : `BiModalEncoder`
    pooled_output_dim : `int`
    fusion_method : `str`, optional (default = `"sum"`)
    dropout : `float`, optional (default = `0.1`)
    label_namespace : `str`, optional (default = `"labels"`)
    is_multilabel: `bool`, optional (default = `False`)
        Whether the output classification is multilabel.
        (i.e., can have multiple correct answers)
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
        is_multilabel: bool = False,
        *,
        ignore_text: bool = False,
        ignore_image: bool = False,
    ) -> None:
        super().__init__(vocab)

        from allennlp.modules.backbones import VilbertBackbone

        self.backbone = VilbertBackbone(
            vocab,
            text_embeddings,
            image_embeddings,
            encoder,
            pooled_output_dim,
            fusion_method,
            dropout,
        )

        num_labels = vocab.get_vocab_size(label_namespace)
        self.label_namespace = label_namespace

        self.classifier = torch.nn.Linear(pooled_output_dim, num_labels)
        self.dropout = torch.nn.Dropout(dropout)

        self.is_multilabel = is_multilabel
        self.ignore_text = ignore_text
        self.ignore_images = ignore_image

    @classmethod
    def from_huggingface_model_name(
        cls,
        vocab: Vocabulary,
        model_name: str,
        image_feature_dim: int,
        image_num_hidden_layers: int,
        image_hidden_size: int,
        image_num_attention_heads: int,
        combined_hidden_size: int,
        combined_num_attention_heads: int,
        pooled_output_dim: int,
        image_intermediate_size: int,
        image_attention_dropout: float,
        image_hidden_dropout: float,
        image_biattention_id: List[int],
        text_biattention_id: List[int],
        text_fixed_layer: int,
        image_fixed_layer: int,
        pooled_dropout: float = 0.1,
        fusion_method: str = "sum",
        *,
        ignore_text: bool = False,
        ignore_image: bool = False,
    ):
        transformer = AutoModel.from_pretrained(model_name)

        # Albert (and maybe others?) has this "embedding_size", that's different from "hidden_size".
        # To get them to the same dimensionality, it uses a linear transform after the embedding
        # layer, which we need to pull out and copy here.
        if hasattr(transformer.config, "embedding_size"):
            config = transformer.config

            text_embeddings = TransformerEmbeddings.from_pretrained_module(
                transformer.embeddings, output_size=config.hidden_dim
            )

            from transformers.models.albert.modeling_albert import AlbertModel

            if isinstance(transformer, AlbertModel):
                text_embeddings.linear_transform = deepcopy(
                    transformer.encoder.embedding_hidden_mapping_in
                )
            else:
                logger.warning(
                    "Unknown model that uses separate embedding size; weights of the linear "
                    f"transform will not be initialized.  Model type is: {transformer.__class__}"
                )
        else:
            text_embeddings = TransformerEmbeddings.from_pretrained_module(transformer.embeddings)

        image_embeddings = ImageFeatureEmbeddings(
            feature_size=image_feature_dim,
            embedding_size=image_hidden_size,
            dropout=image_hidden_dropout,
        )

        encoder = BiModalEncoder.from_pretrained_module(
            pretrained_module=transformer,
            num_hidden_layers2=image_num_hidden_layers,
            hidden_size2=image_hidden_size,
            num_attention_heads2=image_num_attention_heads,
            combined_hidden_size=combined_hidden_size,
            combined_num_attention_heads=combined_num_attention_heads,
            intermediate_size2=image_intermediate_size,
            attention_dropout2=image_attention_dropout,
            hidden_dropout2=image_hidden_dropout,
            biattention_id1=text_biattention_id,
            biattention_id2=image_biattention_id,
            fixed_layer1=text_fixed_layer,
            fixed_layer2=image_fixed_layer,
        )
        return cls(
            vocab=vocab,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            encoder=encoder,
            pooled_output_dim=pooled_output_dim,
            fusion_method=fusion_method,
            dropout=pooled_dropout,
            ignore_text=ignore_text,
            ignore_image=ignore_image,
        )

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        text: TextFieldTensors,
        labels: Optional[torch.Tensor] = None,
        label_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        box_features : `Tensor`
            Shape: `(batch_size, num_boxes, feature_size)`

        box_coordinates : `Tensor`
            Shape: `(batch_size, num_boxes, 4)`

        box_mask : `Tensor`
            A bool and 0-1 tensor of shape `(batch_size, num_boxes)`.

        text : `TextFieldTensors`

        label : `Optional[Tensor]`

        label_weights : `Optional[Tensor]`

        """

        batch_size = box_features.size(0)

        if self.ignore_images:
            box_features = torch.zeros_like(box_features)
            box_coordinates = torch.zeros_like(box_coordinates)
            box_coordinates[..., 2] = 1
            box_coordinates[..., 3] = 1
            box_mask = torch.ones_like(box_mask)

        if self.ignore_text:
            dummy_text = {}
            for embedder_name, tensor_dict in text.items():
                dummy_tensor_dict = {}
                for tensor_name, tensor in tensor_dict.items():
                    if "mask" in tensor_name:
                        tensor = torch.ones_like(tensor)
                    else:
                        tensor = torch.zeros_like(tensor)
                    dummy_tensor_dict[tensor_name] = tensor
                dummy_text[embedder_name] = dummy_tensor_dict
            text = dummy_text

        backbone_outputs = self.backbone(box_features, box_coordinates, box_mask, text)

        # Shape: (batch_size, num_labels)
        logits = self.classifier(backbone_outputs["pooled_boxes_and_text"])

        # Shape: (batch_size, num_labels)
        if self.is_multilabel:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        outputs = {"logits": logits, "probs": probs}
        outputs = self._compute_loss_and_metrics(batch_size, outputs, labels, label_weights)

        return outputs

    def _compute_loss_and_metrics(
        self,
        batch_size: int,
        outputs: torch.Tensor,
        label: torch.Tensor,
        label_weights: Optional[torch.Tensor] = None,
    ):
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
