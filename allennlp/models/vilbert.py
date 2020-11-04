import collections
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy


from allennlp.modules.transformer import (
    TextEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
    TransformerPooler,
    TransformerModule,
)

logger = logging.getLogger(__name__)


@Model.register("nlvr2_vilbert")
@Model.register("nlvr2_vilbert_from_pretrained", constructor="from_pretrained_module")
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
        text_embeddings: TextEmbeddings,
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

        self.t_pooler = TimeDistributed(TransformerPooler(encoder.hidden_size1, pooled_output_dim))
        self.v_pooler = TimeDistributed(TransformerPooler(encoder.hidden_size2, pooled_output_dim))
        self.classifier = torch.nn.Linear(pooled_output_dim * 2, 2)
        self.dropout = torch.nn.Dropout(dropout)

    @classmethod
    def from_pretrained_module(  # type: ignore
        cls,
        pretrained_module: Union[str, torch.nn.Module],
        vocab: Vocabulary,
        image_num_hidden_layers: int,
        image_feature_dim: int,
        image_hidden_size: int,
        combined_hidden_size: int,
        pooled_output_dim: int,
        image_intermediate_size: int,
        image_num_attention_heads: int,
        combined_num_attention_heads: int,
        image_attention_dropout: float,
        image_hidden_dropout: float,
        v_biattention_id: List[int],
        t_biattention_id: List[int],
        fixed_t_layer: int,
        fixed_v_layer: int,
        pooled_dropout: float = 0.1,
        fusion_method: str = "sum",
        fast_mode: bool = False,
        with_coattention: bool = True,
        in_batch_pairs: bool = False,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        # **kwargs,
    ):
        pretrained_module = cls.get_relevant_module(
            pretrained_module, source=source, mapping=mapping
        )
        submodules = cls._get_mapped_submodules(pretrained_module, source, mapping)
        text_embeddings = TextEmbeddings.get_relevant_module(submodules["embeddings"])

        if "huggingface" in source:
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
            feature_dim=image_feature_dim,
            hidden_dim=image_hidden_size,
            dropout=image_hidden_dropout,
        )

        encoder = BiModalEncoder.from_pretrained_module(
            pretrained_module=pretrained_module.encoder,
            source=source,
            num_hidden_layers2=image_num_hidden_layers,
            hidden_size2=image_hidden_size,
            combined_hidden_size=combined_hidden_size,
            intermediate_size2=image_intermediate_size,
            num_attention_heads2=image_num_attention_heads,
            combined_num_attention_heads=combined_num_attention_heads,
            attention_dropout2=image_attention_dropout,
            hidden_dropout2=image_hidden_dropout,
            biattention_id2=v_biattention_id,
            biattention_id1=t_biattention_id,
            fixed_layer1=fixed_t_layer,
            fixed_layer2=fixed_v_layer,
            fast_mode=fast_mode,
            with_coattention=with_coattention,
            in_batch_pairs=in_batch_pairs,
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

    def consistency(self, reset: bool) -> float:
        num_consistent_groups = sum(1 for c in self.consistency_wrong_map.values() if c == 0)
        value = float(num_consistent_groups) / len(self.consistency_wrong_map)
        if reset:
            self.consistency_wrong_map.clear()
        return value

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        identifier: List[str],
        sentence: TextFieldTensors,
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        batch_size, num_images, _, feature_size = box_features.size()

        # TODO(mattg): have this make fewer assumptions.
        input_ids = sentence["tokens"]["token_ids"]
        token_type_ids = sentence["tokens"]["type_ids"]
        attention_mask = sentence["tokens"]["mask"]

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
        v_embedding_output = self.image_embeddings(box_features, box_coordinates.float())
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
        if label is not None:
            outputs["loss"] = self.loss(logits, label).sum()
            self._denotation_accuracy(logits, label)
            # Update group predictions for consistency computation
            predicted_binary = logits.argmax(1)
            for i in range(len(identifier)):
                ident_parts = identifier[i].split("-")
                group_id = "-".join([ident_parts[0], ident_parts[1], ident_parts[-1]])
                self.consistency_wrong_map.setdefault(group_id, 0)
                if predicted_binary[i].item() != label[i].item():
                    self.consistency_wrong_map[group_id] += 1
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "denotation_acc": self._denotation_accuracy.get_metric(reset),
            "consistency": self.consistency(reset),
        }
