from typing import List, Optional, Dict

import torch

from allennlp.common import FromParams

from allennlp.modules.transformer.transformer_module import TransformerModule


class TextEmbeddings(TransformerModule, FromParams):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        dropout: float,
    ):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ImageFeatureEmbeddings(torch.nn.Module, FromParams):
    """Construct the embeddings from image, spatial location (omit now) and
    token_type embeddings.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        self.image_embeddings = torch.nn.Linear(feature_dim, hidden_dim)
        self.image_location_embeddings = torch.nn.Linear(4, hidden_dim)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, image_feature: torch.Tensor, image_location: torch.Tensor):
        img_embeddings = self.image_embeddings(image_feature)
        loc_embeddings = self.image_location_embeddings(image_location)
        embeddings = self.layer_norm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Embeddings(TransformerModule, FromParams):
    """
    General class for embeddings for any modality.
    """

    def __init__(self, embeddings: List[torch.nn.Module], hidden_size: int, dropout: float):
        super().__init__()
        self.embeddings = embeddings
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == len(self.embeddings)
        outputs = []
        for i, inp in enumerate(inputs):
            outputs.append(self.embeddings[i](inp))

        outputs = sum(outputs)  # type: ignore
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)
        return outputs


class TransformerEmbeddings(TransformerModule, FromParams):
    """
    Construct the embeddings from word, position and token_type embeddings.
    Details in the paper:
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al, 2019]
    (https://api.semanticscholar.org/CorpusID:52967399)
    """

    _relevant_module = "embeddings"
    _huggingface_mapping = {"LayerNorm": "embeddings.layer_norm", "dropout": "embeddings.dropout"}

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int = 0,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)
        token_type_embeddings = torch.nn.Embedding(type_vocab_size, hidden_size)

        embeddings = [word_embeddings, position_embeddings, token_type_embeddings]

        self.embeddings = Embeddings(embeddings, hidden_size, dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        input_shape = input_ids.size()
        device = input_ids.device
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embeddings = self.embeddings([input_ids, position_ids, token_type_ids])

        return embeddings

    @classmethod
    def _get_input_arguments(
        cls,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        submodules = cls._get_mapped_submodules(pretrained_module, source, mapping)

        final_kwargs = {}

        final_kwargs["vocab_size"] = submodules["word_embeddings"].num_embeddings
        final_kwargs["hidden_size"] = submodules["word_embeddings"].embedding_dim
        final_kwargs["pad_token_id"] = submodules["word_embeddings"].padding_idx
        final_kwargs["max_position_embeddings"] = submodules["position_embeddings"].num_embeddings
        final_kwargs["type_vocab_size"] = submodules["token_type_embeddings"].num_embeddings

        final_kwargs.update(**kwargs)

        return final_kwargs
