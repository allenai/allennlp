import logging
from typing import Dict, List

import torch
from overrides import overrides
from math import prod

from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.backbones.backbone import Backbone
from allennlp.modules.transformer import (
    BiModalEncoder,
    ImageFeatureEmbeddings,
    TransformerEmbeddings,
    TransformerPooler,
)

logger = logging.getLogger(__name__)


@Backbone.register("vilbert")
@Backbone.register("vilbert_from_huggingface", constructor="from_huggingface_model_name")
class VilbertBackbone(Backbone):
    """
    Uses a Vilbert model as a `Backbone`.
    Registered as a `Backbone` with name "vilbert".
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
        vocab_namespace: str = "tokens",
    ) -> None:
        super().__init__()
        self.fusion_method = fusion_method
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.encoder = encoder

        self.t_pooler = TransformerPooler(encoder.hidden_size1, pooled_output_dim)
        self.v_pooler = TransformerPooler(encoder.hidden_size2, pooled_output_dim)
        self.dropout = torch.nn.Dropout(dropout)

        self._vocab = vocab
        self._namespace = vocab_namespace

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
        fusion_method: str = "sum",
    ):
        text_embeddings = TransformerEmbeddings.from_pretrained_module(model_name)

        image_embeddings = ImageFeatureEmbeddings(
            feature_size=image_feature_dim,
            embedding_size=image_hidden_size,
            dropout=image_hidden_dropout,
        )

        encoder = BiModalEncoder.from_pretrained_module(
            model_name,
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
        )

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        text: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:
        if "token_ids" in text["tokens"]:
            token_ids = text["tokens"]["token_ids"]
        else:
            token_ids = text["tokens"]["tokens"]

        # Shape: (batch_size, num_tokens)
        token_type_ids = text["tokens"].get("type_ids")
        # Shape: (batch_size, num_tokens)
        attention_mask = text["tokens"].get("mask")

        # TODO: clean this up
        change_dimensions = False
        if len(box_features.shape) > 3:
            change_dimensions = True
            sizes = box_features.shape
            batch_size = sizes[0]
            num_boxes = sizes[-2]
            feature_size = sizes[-1]
            other_dimensions = sizes[1:-2]
            token_ids = token_ids.repeat_interleave(prod(other_dimensions), 0)
            token_type_ids = token_type_ids.repeat_interleave(prod(other_dimensions), 0)
            attention_mask = attention_mask.repeat_interleave(prod(other_dimensions), 0)

            # box_features = box_features.view(batch_size * num_images, num_boxes, feature_dimension)
            # box_coordinates = box_coordinates.view(
            #     batch_size * num_images, box_coordinates.shape[2], box_coordinates.shape[3]
            # )
            # box_mask = box_mask.view(batch_size * num_images, box_mask.shape[2])
            box_features = box_features.view(
                batch_size * prod(other_dimensions), num_boxes, feature_size
            )
            box_coordinates = box_coordinates.view(
                batch_size * prod(other_dimensions),
                box_coordinates.shape[2],
                box_coordinates.shape[3],
            )
            box_mask = box_mask.view(batch_size * prod(other_dimensions), box_mask.shape[2])
        else:
            batch_size, _, feature_size = box_features.size()
        # end TODO: clean this up

        # Shape: (batch_size *  prod(other_dimensions), num_tokens, embedding_dim)
        embedding_output = self.text_embeddings(token_ids, token_type_ids)
        num_tokens = embedding_output.size(1)

        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        if attention_mask is not None:
            extended_attention_mask = attention_mask
        else:
            extended_attention_mask = None

        extended_image_attention_mask = box_mask

        # Shape: (batch_size * prod(other_dimensions), feature_size, num_tokens)
        # TODO (epwalsh): Why all zeros?? This doesn't seem right.
        extended_co_attention_mask = torch.zeros(
            batch_size * prod(other_dimensions),
            feature_size,
            num_tokens,
            dtype=extended_image_attention_mask.dtype,
        )

        # Shape: (batch_size * prod(other_dimensions), num_boxes, image_embedding_dim)
        v_embedding_output = self.image_embeddings(box_features, box_coordinates)

        encoded_layers_t, encoded_layers_v = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
        )

        # Shape: (batch_size * prod(other_dimensions), num_tokens, embedding_dim)
        sequence_output_t = encoded_layers_t[:, :, :, -1]
        # Shape: (batch_size * prod(other_dimensions), num_boxes, image_embedding_dim)
        sequence_output_v = encoded_layers_v[:, :, :, -1]

        # Shape: (batch_size * prod(other_dimensions), pooled_output_dim)
        pooled_output_t = self.t_pooler(sequence_output_t)
        # Shape: (batch_size * prod(other_dimensions), pooled_output_dim)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if change_dimensions:
            some_dimensions = [batch_size] + list(other_dimensions)
            sequence_output_t.view(
                some_dimensions + [sequence_output_t.shape[-2], sequence_output_t.shape[-1]]
            )
            sequence_output_v.view(
                some_dimensions + [sequence_output_v.shape[-2], sequence_output_v.shape[-1]]
            )
            pooled_output_t.view(
                some_dimensions + [pooled_output_t.shape[-1]]
            )
            pooled_output_v.view(
                some_dimensions + [pooled_output_v.shape[-1]]
            )

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise ValueError(f"Fusion method '{self.fusion_method}' not supported")

        return {
            "encoded_boxes": sequence_output_v,
            "encoded_boxes_mask": box_mask,
            "encoded_boxes_pooled": pooled_output_v,
            "encoded_text": sequence_output_t,
            "encoded_text_mask": attention_mask,
            "encoded_text_pooled": pooled_output_t,
            "pooled_boxes_and_text": pooled_output,
        }
