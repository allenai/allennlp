from typing import Optional, TYPE_CHECKING

import torch

from allennlp.common import FromParams
from allennlp.modules.transformer.layer_norm import LayerNorm
from allennlp.modules.transformer.transformer_module import TransformerModule

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


class Embeddings(TransformerModule, FromParams):
    """
    General class for embeddings for any modality.

    # Parameters

    embeddings : `torch.nn.ModuleDict`
        Named embedding layers. Eg. `"word_embeddings"`, `"position_embeddings"`, etc.
        All the embedding layers are expected to have different inputs; the output
        of one will not be passed to the other. All the layers should have the same
        `embedding_dim`/`out_features`.
    embedding_size : `int`
        The `embedding_dim` of all the embedding layers.
    dropout : `float`
        The probability of an element to be zeroed.
    """

    def __init__(
        self,
        embeddings: torch.nn.ModuleDict,
        embedding_size: int,
        dropout: float,
        layer_norm_eps: float = 1e-12,  # different from Huggingface!
    ):
        super().__init__()
        for name, embedding_layer in embeddings.named_children():
            if isinstance(embedding_layer, torch.nn.Embedding):
                assert embedding_layer.embedding_dim == embedding_size
            elif isinstance(embedding_layer, torch.nn.Linear):
                assert embedding_layer.out_features == embedding_size
            else:
                raise TypeError(
                    'Layer "{}" must be of type `torch.nn.Embedding` or `torch.nn.Linear`.'.format(
                        name
                    )
                )
        self.embeddings = embeddings
        self.layer_norm = LayerNorm(embedding_size, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, *inputs) -> torch.Tensor:
        assert len(inputs) == len(self.embeddings)
        outputs = []
        for i, layer in enumerate(self.embeddings.children()):
            outputs.append(layer(inputs[i]))

        outputs = sum(outputs)  # type: ignore
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)
        return outputs


class ImageFeatureEmbeddings(Embeddings):
    """
    Embedding module for image features.

    # Parameters

    feature_size : `int`
        Number of image features.
    embedding_size : `int`
        The `embedding_dim` of all the embedding layers.
    dropout : `float` (default = `0.0`)
        The probability of an element to be zeroed.
    """

    def __init__(self, feature_size: int, embedding_size: int, dropout: float = 0.0):
        image_embeddings = torch.nn.Linear(feature_size, embedding_size)
        location_embeddings = torch.nn.Linear(4, embedding_size, bias=False)
        embeddings = torch.nn.ModuleDict(
            {"image_embeddings": image_embeddings, "location_embeddings": location_embeddings}
        )
        super().__init__(embeddings, embedding_size, dropout)


class TransformerEmbeddings(Embeddings):
    """
    Construct the embeddings from word, position and token_type embeddings.
    Details in the paper:
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al, 2019]
    (https://api.semanticscholar.org/CorpusID:52967399)

    # Parameters

    vocab_size : `int`
        The size of the input vocab.
    embedding_size : `int`
        The `embedding_dim` of all the embedding layers.
    pad_token_id : `int` (default = `0`)
        The token id of the `<pad>` token.
    max_position_embeddings : `int` (default = `512`)
        The maximum number of positions.
    type_vocab_size : `int` (default = `2`)
        The size of the input token_type vocab.
    dropout : `int` (default = `0.1`)
        The probability of an element to be zeroed.
    output_size : `int`, optional (default = `None`)
        Optionally apply a linear transform after the dropout, projecting to `output_size`.
    """

    _pretrained_relevant_module = ["embeddings", "bert.embeddings", "roberta.embeddings"]
    _pretrained_mapping = {
        "LayerNorm": "layer_norm",
        "word_embeddings": "embeddings.word_embeddings",
        "position_embeddings": "embeddings.position_embeddings",
        "token_type_embeddings": "embeddings.token_type_embeddings",
        # Albert is a special case. A linear projection is applied to the embeddings,
        # but that linear transformation lives in the encoder.
        "albert.embeddings.LayerNorm": "layer_norm",
        "albert.embeddings.word_embeddings": "embeddings.word_embeddings",
        "albert.embeddings.position_embeddings": "embeddings.position_embeddings",
        "albert.embeddings.token_type_embeddings": "embeddings.token_type_embeddings",
        "albert.encoder.embedding_hidden_mapping_in": "linear_transform",
    }
    _pretrained_ignore = [
        # Ignore these for Albert case.
        r"^albert\.pooler\..*",
        r"^albert\.encoder\.albert_layer_groups\..*",
        r"^predictions\.*",
    ]

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        pad_token_id: int = 0,
        max_position_embeddings: int = 512,
        position_pad_token_id: Optional[int] = None,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,  # different from Huggingface!
        output_size: Optional[int] = None,
    ):
        embedding_dict = {}

        word_embeddings = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        embedding_dict["word_embeddings"] = word_embeddings

        if max_position_embeddings > 0:
            position_embeddings = torch.nn.Embedding(
                max_position_embeddings, embedding_size, padding_idx=position_pad_token_id
            )
            embedding_dict["position_embeddings"] = position_embeddings

        if type_vocab_size > 0:
            token_type_embeddings = torch.nn.Embedding(type_vocab_size, embedding_size)
            embedding_dict["token_type_embeddings"] = token_type_embeddings

        embeddings = torch.nn.ModuleDict(embedding_dict)

        super().__init__(embeddings, embedding_size, dropout, layer_norm_eps=layer_norm_eps)

        # For Albert, the embedding size is different than the hidden size used
        # in the model, so a linear transform is applied.
        if output_size:
            self.linear_transform = torch.nn.Linear(embedding_size, output_size)

    def forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        # Parameters
        input_ids : `torch.Tensor`
            Shape `batch_size x seq_len`
        attention_mask : `torch.Tensor`
            Shape `batch_size x seq_len`. This parameter is ignored, but it is here for compatibility.
        token_type_ids : `torch.Tensor`, optional
            Shape `batch_size x seq_len`
        position_ids : `torch.Tensor`, optional
            Shape `batch_size x seq_len`
        """

        input_shape = input_ids.size()
        device = input_ids.device
        seq_length = input_shape[1]

        embedding_inputs = [input_ids]

        if attention_mask is None:
            attention_mask = input_ids != self.embeddings["word_embeddings"].padding_idx

        if "position_embeddings" in self.embeddings:
            if position_ids is None:
                padding_idx = self.embeddings["position_embeddings"].padding_idx
                if padding_idx is None:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(input_shape)
                else:
                    # In the RoBERTa case, position indices start with padding_idx + 1. Also, RoBERTa likes
                    # to respect padding in its position ids.
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=device) + 1
                    position_ids = position_ids.unsqueeze(0).expand(input_shape) * attention_mask
                    position_ids += padding_idx
            embedding_inputs.append(position_ids)

        if "token_type_embeddings" in self.embeddings:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            embedding_inputs.append(token_type_ids)

        embeddings = super().forward(*embedding_inputs)

        if hasattr(self, "linear_transform"):
            embeddings = self.linear_transform(embeddings)

        return embeddings

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs):
        final_kwargs = {
            "vocab_size": config.vocab_size,
            "pad_token_id": config.pad_token_id,
            "max_position_embeddings": config.max_position_embeddings,
            "type_vocab_size": config.type_vocab_size,
            "layer_norm_eps": config.layer_norm_eps,
        }
        # For Albert, the embedding size is different than the hidden size used
        # in the model, so a linear transform is applied.
        if hasattr(config, "embedding_size"):
            final_kwargs["embedding_size"] = config.embedding_size
            final_kwargs["output_size"] = config.hidden_size
        else:
            final_kwargs["embedding_size"] = config.hidden_size
        if config.model_type == "roberta":
            final_kwargs["position_pad_token_id"] = config.pad_token_id
        final_kwargs.update(**kwargs)
        return cls(**final_kwargs)
