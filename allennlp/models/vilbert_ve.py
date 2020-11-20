import logging
from copy import deepcopy
from typing import Dict, List

from overrides import overrides
import numpy as np
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import TextEmbeddings, ImageFeatureEmbeddings, BiModalEncoder
from allennlp.training.metrics import CategoricalAccuracy

from transformers.modeling_auto import AutoModel

logger = logging.getLogger(__name__)


@Model.register("ve_vilbert")
@Model.register("ve_vilbert_from_huggingface", constructor="from_huggingface_model_name")
class VEVilbert(Model):
    """
    Model for VE task based on the VilBERT paper.
    # Parameters
    vocab : `Vocabulary`
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
        from allennlp.modules.transformer import ActivationLayer

        super().__init__(vocab)

        self.fusion_method = fusion_method

        self.embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.encoder = encoder

        self.t_pooler = ActivationLayer(
            encoder.hidden_size1, pooled_output_dim, torch.nn.ReLU(), pool=True
        )
        self.v_pooler = ActivationLayer(
            encoder.hidden_size2, pooled_output_dim, torch.nn.ReLU(), pool=True
        )

        num_labels = vocab.get_vocab_size(label_namespace)
        self.label_namespace = label_namespace

        self.classifier = torch.nn.Linear(pooled_output_dim, num_labels)
        self.dropout = torch.nn.Dropout(dropout)

        self.accuracy = CategoricalAccuracy()

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
        v_biattention_id: List[int],
        t_biattention_id: List[int],
        fixed_t_layer: int,
        fixed_v_layer: int,
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
            biattention_id1=t_biattention_id,
            biattention_id2=v_biattention_id,
            fixed_layer1=fixed_t_layer,
            fixed_layer2=fixed_v_layer,
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
        sentence1: TextFieldTensors,
        sentence2: TextFieldTensors,
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        #TODO: change to these references.
        sentence1 - premise
        sentence2 - hypothesis
        """

        batch_size, _, feature_size = box_features.size()

        # Already tokenized.
        # !! This currently only works with the default pretrainedtokenizer.

        if "token_ids" in sentence1["tokens"]:
            sentence1_token_ids = sentence1["tokens"]["token_ids"]
        else:
            sentence1_token_ids = sentence1["tokens"]["tokens"]

        sentence1_token_type_ids = sentence1["tokens"].get("type_ids")
        # !!!Note: does no mask mean we need to deal with it?
        sentence1_attention_mask = sentence1["tokens"].get("mask")

        if "token_ids" in sentence2["tokens"]:
            sentence2_token_ids = sentence2["tokens"]["token_ids"]
        else:
            sentence2_token_ids = sentence2["tokens"]["tokens"]

        sentence2_token_type_ids = sentence2["tokens"].get("type_ids")
        sentence2_attention_mask = sentence2["tokens"].get("mask")

        # All batch instances will always have the same number of images and boxes, so no masking
        # is necessary, and this is just a tensor of ones.
        image_attention_mask = torch.ones_like(box_coordinates[:, :, 0])

        # (batch_size, num_tokens, embedding_dim)
        embedding_sentence1_output = self.embeddings(sentence1_token_ids, sentence1_token_type_ids)
        num_tokens1 = embedding_sentence1_output.size(1)

        embedding_sentence2_output = self.embeddings(sentence2_token_ids, sentence2_token_type_ids)
        num_tokens2 = embedding_sentence2_output.size(1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        if sentence1_attention_mask is not None:
            extended_attention_mask1 = (
                sentence1_attention_mask.unsqueeze(1).unsqueeze(2).float().log()
            )
        else:
            extended_attention_mask1 = None
        if sentence2_attention_mask is not None:
            extended_attention_mask2 = (
                sentence2_attention_mask.unsqueeze(1).unsqueeze(2).float().log()
            )
        else:
            extended_attention_mask2 = None

        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2).float().log()

        extended_co_attention_mask = torch.zeros(
            batch_size,
            feature_size,
            num_tokens1 + num_tokens2,  # they will be aggregated.
            dtype=extended_image_attention_mask.dtype,
        )

        # aggregate the 2 pieces of text. #dim: num_tokens one.
        embedding_output = torch.cat(
            [embedding_sentence1_output, embedding_sentence2_output], dim=1
        )
        if extended_attention_mask1 is not None and extended_attention_mask2 is not None:
            extended_attention_mask = torch.cat(
                [extended_attention_mask1, extended_attention_mask2], dim=-1
            )
        else:
            extended_attention_mask = None  # 0#torch.zeros(embedding_output.shape)

        # (batch_size, num_boxes, image_embedding_dim)
        v_embedding_output = self.image_embeddings(box_features, box_coordinates)

        encoded_layers_t, encoded_layers_v = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
        )

        sequence_output_t = encoded_layers_t[:, :, :, -1]
        sequence_output_v = encoded_layers_v[:, :, :, -1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise ValueError(f"Fusion method '{self.fusion_method}' not supported")

        logits = self.classifier(pooled_output)
        probs = torch.sigmoid(logits)

        outputs = {"logits": logits, "probs": probs}
        if label is not None:
            outputs["loss"] = torch.nn.functional.cross_entropy(logits, label)
            self.accuracy(logits, label)
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
