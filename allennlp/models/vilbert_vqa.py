import collections
import logging
from copy import deepcopy
from typing import Dict, List

from overrides import overrides
import torch

from allennlp.common import FromParams
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import TextEmbeddings, ImageFeatureEmbeddings, BiModalEncoder
from allennlp.nn import util
from allennlp.training.metrics import F1MultiLabelMeasure

from transformers.modeling_auto import AutoModel

logger = logging.getLogger(__name__)


@Model.register("vqa_vilbert")
@Model.register("vqa_vilbert_from_huggingface", constructor="from_huggingface_model_name")
class VqaVilbert(Model):
    """
    Model for VQA task based on the VilBERT paper.

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
        label_namespace: str = "answers",
    ) -> None:
        from allennlp.modules import TimeDistributed
        from allennlp.modules.transformer import ActivationLayer

        super().__init__(vocab)
        self.loss = torch.nn.BCELoss()
        self.consistency_wrong_map: Dict[str, int] = collections.Counter()
        self.f1 = F1MultiLabelMeasure(average="micro")
        self.fusion_method = fusion_method

        self.embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.encoder = encoder

        self.v_pooler = ActivationLayer(encoder.hidden_size1, pooled_output_dim, torch.nn.ReLU())
        self.t_pooler = ActivationLayer(encoder.hidden_size2, pooled_output_dim, torch.nn.ReLU())

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
        v_biattention_id: List[int],
        t_biattention_id: List[int],
        fixed_t_layer: int,
        fixed_v_layer: int,
        pooled_dropout: float = 0.1,
        fusion_method: str = "sum",
    ):
        transformer = AutoModel.from_pretrained(model_name)

        # TODO(mattg): This call to `transformer.embeddings` works with some transformers, but I'm
        # not sure it works for all of them, or what to do if it fails.
        # We should probably pull everything up until the instantiation of the image feature
        # embedding out into a central "transformers_util" module, or something, and just have a
        # method that pulls an initialized embedding layer out of a huggingface model.  One place
        # for this somewhat hacky code to live, instead of having to duplicate it in various models.
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

        encoder = BiModalEncoder.from_huggingface_model(
            model=transformer,
            image_num_hidden_layers=image_num_hidden_layers,
            image_hidden_size=image_hidden_size,
            image_num_attenton_heads=image_num_attention_heads,
            combined_hidden_size=combined_hidden_size,
            combined_num_attention_heads=combined_num_attention_heads,
            image_intermediate_size=image_intermediate_size,
            image_attention_dropout=image_attention_dropout,
            image_hidden_dropout=image_hidden_dropout,
            v_biattention_id=v_biattention_id,
            t_biattention_id=t_biattention_id,
            fixed_t_layer=fixed_t_layer,
            fixed_v_layer=fixed_v_layer,
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
        question: TextFieldTensors,
        labels: torch.Tensor = None,
        label_weights: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        batch_size, _, feature_size = box_features.size()

        # TODO(mattg): have this make fewer assumptions.
        input_ids = question["tokens"]["token_ids"]
        token_type_ids = question["tokens"]["type_ids"]
        attention_mask = question["tokens"]["mask"]

        # All batch instances will always have the same number of images and boxes, so no masking
        # is necessary, and this is just a tensor of ones.
        image_attention_mask = torch.ones_like(box_coordinates[:, :, 0])

        # (batch_size, num_tokens, embedding_dim)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        num_tokens = embedding_output.size(1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float().log()
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2).float().log()

        # TODO(matt): it looks like the co-attention logic is all currently commented out; not sure
        # that this is necessary.
        extended_co_attention_mask = torch.zeros(
            batch_size,
            feature_size,
            num_tokens,
            dtype=extended_image_attention_mask.dtype,
        )

        # (batch_size, num_boxes, image_embedding_dim)
        v_embedding_output = self.image_embeddings(box_features, box_coordinates)
        
        encoded_layers_v, encoded_layers_t = self.encoder(
            v_embedding_output,
            embedding_output,
            extended_image_attention_mask,
            extended_attention_mask,
            extended_co_attention_mask,
        )
        
        sequence_output_t = encoded_layers_t[:, :, :, -1]
        sequence_output_v = encoded_layers_v[:, :, :, -1]

        pooled_output_t = self.t_pooler(sequence_output_t, pool=True)
        pooled_output_v = self.v_pooler(sequence_output_v, pool=True)

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise ValueError(f"Fusion method '{self.fusion_method}' not supported")

        logits = self.classifier(pooled_output)
        probs = torch.sigmoid(logits)

        outputs = {"logits": logits, "probs": probs}
        if labels is not None:
            label_mask = labels > 1  # 0 is padding, 1 is OOV, which we want to ignore

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
            
            outputs["loss"] = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, weighted_labels, weight=binary_label_mask, reduction='sum'
            ) / batch_size
            
            # TODO(mattg): Multi-label F1 is good, but it is not exactly the VQA accuracy metric.
            # That still needs to be implemented.
            self.f1(logits, weighted_labels, binary_label_mask.bool())
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.f1.get_metric(reset)

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
