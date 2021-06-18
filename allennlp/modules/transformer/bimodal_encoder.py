from typing import Optional, List, TYPE_CHECKING

import torch

from allennlp.common import FromParams
from allennlp.modules.util import replicate_layers
from allennlp.modules.transformer.transformer_layer import TransformerLayer
from allennlp.modules.transformer.bimodal_connection_layer import BiModalConnectionLayer
from allennlp.modules.transformer.transformer_module import TransformerModule

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


class BiModalEncoder(TransformerModule, FromParams):
    """
    This module encodes two modalities separately, and performs bi-directional
    attention using a connection layer. It is based on the modified BertEncoder in
    the paper: [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations
    for Vision-and-Language Tasks](https://api.semanticscholar.org/CorpusID:199453025)

    # Parameters

    num_hidden_layers1: `int` (default = `12`)
        Number of hidden layers in the transformer block for the first modality.
    num_hidden_layers2: `int` (default = `12`)
        Number of hidden layers in the transformer block for the second modality.
    hidden_size1: `int` (default = `1024`)
    hidden_size2: `int` (default = `1024`)
    combined_hidden_size: `int` (default = `1024`)
        Hidden size for the connection layer.
    intermediate_size1: `int` (default = `1024`)
    intermediate_size2: `int` (default = `1024`)
    num_attention_heads1: `int` (default = `8`)
    num_attention_heads2: `int` (default = `8`)
    combined_num_attention_heads: `int` (default = `8`)
        Number of attention heads in the connection layer.
    attention_dropout1: `float` (default = `0.1`)
    hidden_dropout1: `float` (default = `0.1`)
    attention_dropout2: `float` (default = `0.1`)
    hidden_dropout2: `float` (default = `0.1`)
    biattention_id1: `List`, optional (default = `[1]`)
    biattention_id2: `List`, optional (default = `[1]`)
    fixed_layer1: `int` (default = `0`)
    fixed_layer2: `int` (default = `0`)
    fast_mode: `bool` (default = `False`)
    with_coattention: `bool` (default = `True`)
    in_batch_pairs: `bool` (default = `False`)
    """

    _pretrained_mapping = {"layer": "layers1"}
    _pretrained_relevant_module = ["encoder", "bert.encoder"]
    _pretrained_allow_missing = [r"^layers2\..*", r"^c_layer\..*"]

    def __init__(
        self,
        num_hidden_layers1: int = 12,
        num_hidden_layers2: int = 12,
        hidden_size1: int = 1024,
        hidden_size2: int = 1024,
        combined_hidden_size: int = 1024,
        intermediate_size1: int = 1024,
        intermediate_size2: int = 1024,
        num_attention_heads1: int = 8,
        num_attention_heads2: int = 8,
        combined_num_attention_heads: int = 8,
        attention_dropout1: float = 0.1,
        hidden_dropout1: float = 0.1,
        attention_dropout2: float = 0.1,
        hidden_dropout2: float = 0.1,
        activation: str = "relu",
        biattention_id1: Optional[List[int]] = None,
        biattention_id2: Optional[List[int]] = None,
        fixed_layer1: int = 0,
        fixed_layer2: int = 0,
        fast_mode: bool = False,
        with_coattention: bool = True,
        in_batch_pairs: bool = False,
    ):
        super().__init__()

        self.FAST_MODE = fast_mode
        self.with_coattention = with_coattention
        self.biattention_id1 = biattention_id1 or [1]
        self.biattention_id2 = biattention_id2 or [1]
        self.in_batch_pairs = in_batch_pairs
        self.fixed_layer1 = fixed_layer1
        self.fixed_layer2 = fixed_layer2
        self.combined_size = combined_hidden_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        layer1 = TransformerLayer(
            hidden_size=hidden_size1,
            intermediate_size=intermediate_size1,
            num_attention_heads=num_attention_heads1,
            attention_dropout=attention_dropout1,
            hidden_dropout=hidden_dropout1,
            activation=activation,
        )
        layer2 = TransformerLayer(
            hidden_size=hidden_size2,
            intermediate_size=intermediate_size2,
            num_attention_heads=num_attention_heads2,
            attention_dropout=attention_dropout2,
            hidden_dropout=hidden_dropout2,
            activation=activation,
        )
        connect_layer = BiModalConnectionLayer(
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            combined_hidden_size=combined_hidden_size,
            intermediate_size1=intermediate_size1,
            intermediate_size2=intermediate_size2,
            num_attention_heads=combined_num_attention_heads,
            dropout1=hidden_dropout1,
            dropout2=hidden_dropout2,
            activation=activation,
        )

        self.layers1 = replicate_layers(layer1, num_hidden_layers1)
        self.layers2 = replicate_layers(layer2, num_hidden_layers2)
        self.c_layer = replicate_layers(connect_layer, len(self.biattention_id2))

    def forward(
        self,
        embedding1,
        embedding2,
        attention_mask1,
        attention_mask2,
        co_attention_mask=None,
        output_all_encoded_layers=True,
    ):
        start1 = 0
        start2 = 0
        count = 0
        all_encoder_layers1 = []
        all_encoder_layers2 = []

        batch_size, num_words, hidden_size1 = embedding1.size()
        _, num_regions, hidden_size2 = embedding2.size()

        for layer_id2, layer_id1 in zip(self.biattention_id2, self.biattention_id1):
            end1 = layer_id1
            end2 = layer_id2

            assert self.fixed_layer1 <= end1
            assert self.fixed_layer2 <= end2

            for idx in range(start1, self.fixed_layer1):
                with torch.no_grad():
                    embedding1 = self.layers1[idx](embedding1, attention_mask1).hidden_states
                    start1 = self.fixed_layer1

            for idx in range(start1, end1):
                embedding1 = self.layers1[idx](embedding1, attention_mask1).hidden_states

            for idx in range(start2, self.fixed_layer2):
                with torch.no_grad():
                    embedding2 = self.layers2[idx](embedding2, attention_mask2).hidden_states
                    start2 = self.fixed_layer2

            for idx in range(start2, end2):
                embedding2 = self.layers2[idx](embedding2, attention_mask2).hidden_states

            if count == 0 and self.in_batch_pairs:
                # new batch size is the batch_size ^2
                embedding2 = (
                    embedding2.unsqueeze(0)
                    .expand(batch_size, batch_size, num_regions, hidden_size2)
                    .contiguous()
                    .view(batch_size * batch_size, num_regions, hidden_size2)
                )
                attention_mask2 = (
                    attention_mask2.unsqueeze(0)
                    .expand(batch_size, batch_size, 1, 1, num_regions)
                    .contiguous()
                    .view(batch_size * batch_size, 1, 1, num_regions)
                )

                embedding1 = (
                    embedding1.unsqueeze(1)
                    .expand(batch_size, batch_size, num_words, hidden_size1)
                    .contiguous()
                    .view(batch_size * batch_size, num_words, hidden_size1)
                )
                attention_mask1 = (
                    attention_mask1.unsqueeze(1)
                    .expand(batch_size, batch_size, 1, 1, num_words)
                    .contiguous()
                    .view(batch_size * batch_size, 1, 1, num_words)
                )
                if co_attention_mask is not None:
                    co_attention_mask = (
                        co_attention_mask.unsqueeze(1)
                        .expand(batch_size, batch_size, 1, num_regions, num_words)
                        .contiguous()
                        .view(batch_size * batch_size, 1, num_regions, num_words)
                    )

            if count == 0 and self.FAST_MODE:
                embedding1 = embedding1.expand(
                    embedding2.size(0),
                    embedding1.size(1),
                    embedding1.size(2),
                )
                attention_mask1 = attention_mask1.expand(
                    embedding2.size(0),
                    attention_mask1.size(1),
                    attention_mask1.size(2),
                    attention_mask1.size(3),
                )

            if self.with_coattention:
                embedding1, embedding2 = self.c_layer[count](
                    embedding1,
                    attention_mask1,
                    embedding2,
                    attention_mask2,
                    co_attention_mask,
                )

            start2 = end2
            start1 = end1
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers1.append(embedding1)
                all_encoder_layers2.append(embedding2)

        for idx in range(start2, len(self.layers2)):
            embedding2 = self.layers2[idx](embedding2, attention_mask2).hidden_states

        for idx in range(start1, len(self.layers1)):
            embedding1 = self.layers1[idx](embedding1, attention_mask1).hidden_states

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers1.append(embedding1)
            all_encoder_layers2.append(embedding2)

        return (
            torch.stack(all_encoder_layers1, dim=-1),
            torch.stack(all_encoder_layers2, dim=-1),
        )

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs):
        final_kwargs = {}
        final_kwargs["num_hidden_layers1"] = config.num_hidden_layers
        final_kwargs["hidden_size1"] = config.hidden_size
        final_kwargs["num_attention_heads1"] = config.num_attention_heads
        final_kwargs["attention_dropout1"] = config.attention_probs_dropout_prob
        final_kwargs["hidden_dropout1"] = config.hidden_dropout_prob
        final_kwargs["intermediate_size1"] = config.intermediate_size
        final_kwargs["activation"] = config.hidden_act
        final_kwargs.update(**kwargs)
        return cls(**final_kwargs)
