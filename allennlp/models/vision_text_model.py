import logging
from copy import deepcopy
from typing import Dict, List, Optional

from overrides import overrides
import numpy as np
import torch

from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import (
    TextEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
    TransformerPooler,
)

from transformers.modeling_auto import AutoModel

logger = logging.getLogger(__name__)


@Model.register("vision_model")
class VisionTextModel(Model):
    """
    `VisionTextModel` takes as input a single text input and a single image input
    to produce some output. Example tasks include visual question-answering, visual
    entailment, etc.

    # Parameters

    vocab : `Vocabulary`
    text_embeddings : `TextEmbeddings`
    image_embeddings : `ImageFeatureEmbeddings`
    encoder : `BiModalEncoder`
    pooled_output_dim : `int`
    fusion_method : `str`, optional (default = `"sum"`)
    dropout : `float`, optional (default = `0.1`)
    label_namespace : `str`, optional (default = `"labels"`)
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

        super().__init__(vocab)

        self.fusion_method = fusion_method

        self.embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.encoder = encoder

        self.t_pooler = TransformerPooler(encoder.hidden_size1, pooled_output_dim)
        self.v_pooler = TransformerPooler(encoder.hidden_size2, pooled_output_dim)

        num_labels = vocab.get_vocab_size(label_namespace)
        self.label_namespace = label_namespace

        self.classifier = torch.nn.Linear(pooled_output_dim, num_labels)
        self.dropout = torch.nn.Dropout(dropout)

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
    ):
        transformer = AutoModel.from_pretrained(model_name)

        text_embeddings = deepcopy(transformer.embeddings)

        # Albert (and maybe others?) has this "embedding_size", that's different from "hidden_size".
        # To get them to the same dimensionality, it uses a linear transform after the embedding
        # layer, which we need to pull out and copy here.
        if hasattr(transformer.config, "embedding_size"):
            config = transformer.config

            from transformers.modeling_albert import AlbertModel

            if isinstance(transformer, AlbertModel):
                linear_transform = deepcopy(transformer.encoder.embedding_hidden_mapping_in)
            else:
                logger.warning(
                    "Unknown model that uses separate embedding size; weights of the linear "
                    f"transform will not be initialized.  Model type is: {transformer.__class__}"
                )
                linear_transform = torch.nn.Linear(config.embedding_dim, config.hidden_dim)

            # We can't just use torch.nn.Sequential here, even though that's basically all this is,
            # because Sequential doesn't accept *inputs, only a single argument.

            class EmbeddingsShim(torch.nn.Module):
                def __init__(self, embeddings: torch.nn.Module, linear_transform: torch.nn.Module):
                    super().__init__()
                    self.linear_transform = linear_transform
                    self.embeddings = embeddings

                def forward(self, *inputs, **kwargs):
                    return self.linear_transform(self.embeddings(*inputs, **kwargs))

            text_embeddings = EmbeddingsShim(text_embeddings, linear_transform)

        image_embeddings = ImageFeatureEmbeddings(
            feature_dim=image_feature_dim,
            hidden_dim=image_hidden_size,
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
        )

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        text: TextFieldTensors,
        label: Optional[torch.Tensor] = None,
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

        batch_size, _, feature_size = box_features.size()

        if "token_ids" in text["tokens"]:
            token_ids = text["tokens"]["token_ids"]
        else:
            token_ids = text["tokens"]["tokens"]

        # Shape: (batch_size, num_tokens)
        token_type_ids = text["tokens"].get("type_ids")
        # Shape: (batch_size, num_tokens)
        attention_mask = text["tokens"].get("mask")

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedding_output = self.embeddings(token_ids, token_type_ids)
        num_tokens = embedding_output.size(1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, num_tokens]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, num_tokens]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        if attention_mask is not None:
            # Shape: (batch_size, 1, 1, num_tokens)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_attention_mask = None

        # Shape: (batch_size, 1, 1, num_boxes)
        extended_image_attention_mask = box_mask.unsqueeze(1).unsqueeze(2)

        # Shape: (batch_size, feature_size, num_tokens)
        extended_co_attention_mask = torch.zeros(
            batch_size,
            feature_size,
            num_tokens,
            dtype=extended_image_attention_mask.dtype,
        )

        # Shape: (batch_size, num_boxes, image_embedding_dim)
        v_embedding_output = self.image_embeddings(box_features, box_coordinates)

        # Shape (encoded_layers_t): (batch_size, num_tokens, embedding_dim, num_layers)
        # Shape (encoded_layers_v): (batch_size, num_boxes, image_embedding_dim, num_layers)
        encoded_layers_t, encoded_layers_v = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
        )

        # Shape: (batch_size, num_tokens, embedding_dim)
        sequence_output_t = encoded_layers_t[:, :, :, -1]
        # Shape: (batch_size, num_boxes, image_embedding_dim)
        sequence_output_v = encoded_layers_v[:, :, :, -1]

        # Shape: (batch_size, pooled_output_dim)
        pooled_output_t = self.t_pooler(sequence_output_t)
        # Shape: (batch_size, pooled_output_dim)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise ValueError(f"Fusion method '{self.fusion_method}' not supported")

        # Shape: (batch_size, num_labels)
        logits = self.classifier(pooled_output)
        # Shape: (batch_size, num_labels)
        probs = torch.sigmoid(logits)

        outputs = {"logits": logits, "probs": probs}
        outputs = self._compute_loss_and_metrics(batch_size, outputs, label, label_weights)

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
