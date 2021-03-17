from typing import Optional, Dict

import torch

from allennlp.common import FromParams

from allennlp.modules.transformer.transformer_module import TransformerModule


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

    def __init__(self, embeddings: torch.nn.ModuleDict, embedding_size: int, dropout: float):
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
        self.layer_norm = torch.nn.LayerNorm(embedding_size, eps=1e-12)
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

    _relevant_module = "embeddings"
    _huggingface_mapping = {
        "LayerNorm": "layer_norm",
        "word_embeddings": "embeddings.word_embeddings",
        "position_embeddings": "embeddings.position_embeddings",
        "token_type_embeddings": "embeddings.token_type_embeddings",
    }

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        pad_token_id: int = 0,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
        output_size: Optional[int] = None,
    ):

        embedding_dict = {}

        word_embeddings = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        embedding_dict["word_embeddings"] = word_embeddings

        position_embeddings = torch.nn.Embedding(max_position_embeddings, embedding_size)
        embedding_dict["position_embeddings"] = position_embeddings

        if type_vocab_size > 0:
            token_type_embeddings = torch.nn.Embedding(type_vocab_size, embedding_size)
            embedding_dict["token_type_embeddings"] = token_type_embeddings

        embeddings = torch.nn.ModuleDict(embedding_dict)

        super().__init__(embeddings, embedding_size, dropout)

        # For Albert, the embedding size is different than the hidden size used
        # in the model, so a linear transform is applied.
        if output_size:
            self.linear_transform = torch.nn.Linear(embedding_size, output_size)

    def forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        input_ids : `torch.Tensor`
            Shape `batch_size x seq_len`
        token_type_ids : `torch.Tensor`, optional
            Shape `batch_size x seq_len`
        position_ids : `torch.Tensor`, optional
            Shape `batch_size x seq_len`
        """

        input_shape = input_ids.size()
        device = input_ids.device
        seq_length = input_shape[1]

        embedding_inputs = [input_ids]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        embedding_inputs.append(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if len(self.embeddings) == 3:
            embedding_inputs.append(token_type_ids)

        embeddings = super().forward(*embedding_inputs)

        if hasattr(self, "linear_transform"):
            embeddings = self.linear_transform(embeddings)

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

        final_kwargs["vocab_size"] = submodules["embeddings.word_embeddings"].num_embeddings
        final_kwargs["embedding_size"] = submodules["embeddings.word_embeddings"].embedding_dim
        final_kwargs["pad_token_id"] = submodules["embeddings.word_embeddings"].padding_idx
        final_kwargs["max_position_embeddings"] = submodules[
            "embeddings.position_embeddings"
        ].num_embeddings

        if "embeddings.token_type_embeddings" in submodules:
            final_kwargs["type_vocab_size"] = submodules[
                "embeddings.token_type_embeddings"
            ].num_embeddings

        else:
            final_kwargs["type_vocab_size"] = 0

        final_kwargs.update(**kwargs)

        return final_kwargs
