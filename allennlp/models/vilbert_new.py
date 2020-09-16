import collections
import logging
from copy import deepcopy
from typing import Dict, List, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy

from allennlp.modules.transformer import (
    Embeddings,
    TransformerEmbeddings,
    BiModalEncoder,
    ActivationLayer,
    TransformerModule,
)

logger = logging.getLogger(__name__)


class ImageFeatureEmbeddings(Embeddings):
    """Construct the embeddings from image, spatial location (omit now) and
    token_type embeddings.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.0):

        image_embeddings = torch.nn.Linear(feature_dim, hidden_dim)
        image_location_embeddings = torch.nn.Linear(4, hidden_dim)
        embeddings = [image_embeddings, image_location_embeddings]
        super().__init__(embeddings, hidden_dim, dropout)


@Model.register("nlvr2_vilbert_new")
@Model.register("nlvr2_vilbert_from_huggingface_new", constructor="from_huggingface_model_name")
class Nlvr2Vilbert(Model, TransformerModule):
    """
    Model for the NLVR2 task based on the LXMERT paper (Tan et al. 2019).
    Parameters
    ----------
    vocab: ``Vocabulary``
    """

    # NOTE: This line is unnecessary and will be removed.
    # It's to showcase the possiblity of addressing Matt's comment:
    #   TODO(mattg): This call to `transformer.embeddings` works with some transformers, but I'm
    #   not sure it works for all of them, or what to do if it fails. ...
    # See `from_pretrained_module` defined below to see how it is used.
    _huggingface_mapping = {"encoder": "encoder"}

    def __init__(
        self,
        vocab: Vocabulary,
        text_embeddings: TransformerEmbeddings,
        image_embeddings: ImageFeatureEmbeddings,
        encoder: BiModalEncoder,
        pooled_output_dim: int,
        fusion_method: str = "sum",
        dropout: float = 0.1,
    ) -> None:
        super().__init__(vocab)
        self.loss = torch.nn.CrossEntropyLoss()
        self.consistency_wrong_map: Dict[str, int] = collections.Counter()
        self._denotation_accuracy = CategoricalAccuracy()
        self.fusion_method = fusion_method

        self.embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.encoder = TimeDistributed(encoder)

        self.t_pooler = TimeDistributed(
            ActivationLayer(encoder.hidden_size1, pooled_output_dim, torch.nn.ReLU())
        )
        self.v_pooler = TimeDistributed(
            ActivationLayer(encoder.hidden_size2, pooled_output_dim, torch.nn.ReLU())
        )
        self.classifier = torch.nn.Linear(pooled_output_dim * 2, 2)
        self.dropout = torch.nn.Dropout(dropout)

    @classmethod
    def from_pretrained_module(
        cls,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        required_kwargs = [
            "vocab",
            "image_feature_dim",
            "image_num_hidden_layers",
            "image_hidden_size",
            "combined_hidden_size",
            "pooled_output_dim",
            "image_intermediate_size",
            "image_attention_dropout",
            "image_hidden_dropout",
            "v_biattention_id",
            "t_biattention_id",
            "fixed_t_layer",
            "fixed_v_layer",
        ]
        for key in required_kwargs:
            assert key in kwargs, "`{}` is a required argument.".format(key)

        kwargs["pooled_dropout"] = kwargs.get("pooled_dropout", 0.1)
        kwargs["fusion_method"] = kwargs.get("fusion_method", "sum")
        kwargs["fast_mode"] = kwargs.get("fast_mode", False)
        kwargs["with_coattention"] = kwargs.get("with_coattention", True)
        kwargs["in_batch_pairs"] = kwargs.get("in_batch_pairs", False)

        submodules = cls._get_mapped_submodules(pretrained_module, source, mapping)
        text_embeddings = deepcopy(submodules["embeddings"])
        # FIX: change to below:
        # text_embeddings = TransformerEmbeddings.from_pretrained_module(submodules["embeddings"])

        if "huggingface" in source:
            # FIX: change this part to use mapping.

            if source == "albert-huggingface":
                linear_transform = deepcopy(submodules["encoder"].embedding_hidden_mapping_in)

                class EmbeddingsShim(torch.nn.Module):
                    def __init__(
                        self, embeddings: torch.nn.Module, linear_transform: torch.nn.Module
                    ):
                        super().__init__()
                        self.linear_transform = linear_transform
                        self.embeddings = embeddings

                    def forward(self, *inputs, **kwargs):
                        return self.linear_transform(self.embeddings(*inputs, **kwargs))

                text_embeddings = EmbeddingsShim(text_embeddings, linear_transform)

        image_embeddings = ImageFeatureEmbeddings(
            feature_dim=kwargs["image_feature_dim"],
            hidden_dim=kwargs["image_hidden_size"],
            dropout=kwargs["image_hidden_dropout"],
        )

        encoder = BiModalEncoder.from_pretrained_module(
            pretrained_module=pretrained_module.encoder,
            source=source,
            num_hidden_layers2=kwargs["image_num_hidden_layers"],
            hidden_size2=kwargs["image_hidden_size"],
            combined_hidden_size=kwargs["combined_hidden_size"],
            intermediate_size2=kwargs["image_intermediate_size"],
            attention_dropout2=kwargs["image_attention_dropout"],
            hidden_dropout2=kwargs["image_hidden_dropout"],
            biattention_id2=kwargs["v_biattention_id"],
            biattention_id1=kwargs["t_biattention_id"],
            fixed_layer1=kwargs["fixed_t_layer"],
            fixed_layer2=kwargs["fixed_v_layer"],
            fast_mode=kwargs["fast_mode"],
            with_coattention=kwargs["with_coattention"],
            in_batch_pairs=kwargs["in_batch_pairs"],
        )

        return cls(
            vocab=kwargs["vocab"],
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            encoder=encoder,
            pooled_output_dim=kwargs["pooled_output_dim"],
            fusion_method=kwargs["fusion_method"],
            dropout=kwargs["pooled_dropout"],
        )

    def consistency(self, reset: bool) -> float:
        num_consistent_groups = sum(1 for c in self.consistency_wrong_map.values() if c == 0)
        value = float(num_consistent_groups) / len(self.consistency_wrong_map)
        if reset:
            self.consistency_wrong_map.clear()
        return value

    @overrides
    def forward(
        self,  # type: ignore
        sentence: List[str],
        visual_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        image_id: List[List[str]],
        identifier: List[str],
        sentence_field: TextFieldTensors,
        denotation: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        batch_size, num_images, _, feature_size = visual_features.size()

        input_ids = sentence_field["tokens"]["token_ids"]
        token_type_ids = sentence_field["tokens"]["type_ids"]
        attention_mask = sentence_field["tokens"]["mask"]
        # All batch instances will always have the same number of images and boxes, so no masking
        # is necessary, and this is just a tensor of ones.
        image_attention_mask = torch.ones_like(box_coordinates[:, :, :, 0])

        # (batch_size, num_tokens, embedding_dim)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        num_tokens = embedding_output.size(1)

        # Repeat the embedding dimension, so that the TimeDistributed works out ok
        embedding_output = embedding_output.unsqueeze(1).expand(-1, 2, -1, -1)
        attention_mask = attention_mask.unsqueeze(1).expand(-1, 2, -1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(2).unsqueeze(3).float().log()
        extended_image_attention_mask = image_attention_mask.unsqueeze(2).unsqueeze(3).float().log()

        # TODO(matt): it looks like the co-attention logic is all currently commented out; not sure
        # that this is necessary.
        extended_co_attention_mask = torch.zeros(
            batch_size,
            num_images,
            1,
            feature_size,
            num_tokens,
            dtype=extended_image_attention_mask.dtype,
        )

        # (batch_size, num_images, num_boxes, image_embedding_dim)
        v_embedding_output = self.image_embeddings(visual_features, box_coordinates)
        encoded_layers_t, encoded_layers_v = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
        )

        sequence_output_t = encoded_layers_t[:, :, :, :, -1]
        sequence_output_v = encoded_layers_v[:, :, :, :, -1]

        pooled_output_t = self.t_pooler(sequence_output_t, pool=True)
        pooled_output_v = self.v_pooler(sequence_output_v, pool=True)

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise ValueError(f"Fusion method '{self.fusion_method}' not supported")

        hidden_dim = pooled_output.size(-1)
        logits = self.classifier(pooled_output.view(batch_size, num_images * hidden_dim))

        outputs = {}
        outputs["logits"] = logits
        if denotation is not None:
            outputs["loss"] = self.loss(logits, denotation).sum()
            self._denotation_accuracy(logits, denotation)
            # Update group predictions for consistency computation
            predicted_binary = logits.argmax(1)
            for i in range(len(identifier)):
                ident_parts = identifier[i].split("-")
                group_id = "-".join([ident_parts[0], ident_parts[1], ident_parts[-1]])
                self.consistency_wrong_map.setdefault(group_id, 0)
                if predicted_binary[i].item() != denotation[i].item():
                    self.consistency_wrong_map[group_id] += 1
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "denotation_acc": self._denotation_accuracy.get_metric(reset),
            "consistency": self.consistency(reset),
        }
