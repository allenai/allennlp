from typing import Optional, Dict, List
import torch

from allennlp.common import FromParams

from allennlp.modules.util import replicate_layers

from allennlp.modules.transformer.transformer_layer import TransformerLayer
from allennlp.modules.transformer.bimodal_connection_layer import BiModalConnectionLayer
from allennlp.modules.transformer.transformer_module import TransformerModule


class BiModalEncoder(TransformerModule, FromParams):

    _huggingface_mapping = {"layers1": "layer"}

    def __init__(
        self,
        num_hidden_layers1: int,
        num_hidden_layers2: int,
        hidden_size1: int,
        hidden_size2: int,
        combined_hidden_size: int,
        intermediate_size1: int,
        intermediate_size2: int,
        num_attention_heads: int,
        attention_dropout1: float,
        hidden_dropout1: float,
        attention_dropout2: float,
        hidden_dropout2: float,
        activation: str,
        biattention_id1: List[int],
        biattention_id2: List[int],
        fixed_layer1: int,
        fixed_layer2: int,
        fast_mode: bool = False,
        with_coattention: bool = True,
        in_batch_pairs: bool = False,
    ):
        super().__init__()

        self.FAST_MODE = fast_mode
        self.with_coattention = with_coattention
        self.biattention_id1 = biattention_id1
        self.biattention_id2 = biattention_id2
        self.in_batch_pairs = in_batch_pairs
        self.fixed_layer1 = fixed_layer1
        self.fixed_layer2 = fixed_layer2
        self.combined_size = combined_hidden_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        layer1 = TransformerLayer(
            hidden_size=hidden_size1,
            intermediate_size=intermediate_size1,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout1,
            hidden_dropout=hidden_dropout1,
            activation=activation,
        )
        layer2 = TransformerLayer(
            hidden_size=hidden_size2,
            intermediate_size=intermediate_size2,
            num_attention_heads=num_attention_heads,
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
            num_attention_heads=num_attention_heads,
            dropout1=hidden_dropout1,
            dropout2=hidden_dropout2,
            activation=activation,
        )

        self.layers1 = replicate_layers(layer1, num_hidden_layers1)
        self.layers2 = replicate_layers(layer2, num_hidden_layers2)
        self.c_layer = replicate_layers(connect_layer, len(biattention_id2))

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

        use_co_attention_mask = False
        for layer_id2, layer_id1 in zip(self.biattention_id2, self.biattention_id1):
            end1 = layer_id1
            end2 = layer_id2

            assert self.fixed_layer1 <= end1
            assert self.fixed_layer2 <= end2

            for idx in range(start1, self.fixed_layer1):
                with torch.no_grad():
                    embedding1 = self.layers1[idx](embedding1, attention_mask1)
                    start1 = self.fixed_layer1

            for idx in range(start1, end1):
                embedding1 = self.layers1[idx](embedding1, attention_mask1)

            for idx in range(start2, self.fixed_layer2):
                with torch.no_grad():
                    embedding2 = self.layers2[idx](embedding2, attention_mask2)
                    start2 = self.fixed_layer2

            for idx in range(start2, end2):
                embedding2 = self.layers2[idx](embedding2, attention_mask2)

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
                    use_co_attention_mask,
                )

            start2 = end2
            start1 = end1
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers1.append(embedding1)
                all_encoder_layers2.append(embedding2)

        for idx in range(start2, len(self.layers2)):
            embedding2 = self.layers2[idx](embedding2, attention_mask2)

        for idx in range(start1, len(self.layers1)):
            embedding1 = self.layers1[idx](embedding1, attention_mask1)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers1.append(embedding1)
            all_encoder_layers2.append(embedding2)

        return (
            torch.stack(all_encoder_layers1, dim=-1),
            torch.stack(all_encoder_layers2, dim=-1),
        )

    @classmethod
    def _get_input_arguments(
        cls,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
    ):
        """
        The `pretrained_module` only supplies one of the modalities.
        """
        submodules = cls._get_mapped_submodules(pretrained_module, source, mapping)

        kwargs = {}

        kwargs["num_hidden_layers1"] = len(submodules["layers1"])

        kwargs["hidden_size1"] = submodules["layers1.0.attention.self.query"].in_features
        kwargs["num_attention_heads"] = submodules["layers1.0.attention.self"].num_attention_heads
        kwargs["attention_dropout1"] = submodules["layers1.0.attention.self.dropout"].p
        kwargs["hidden_dropout1"] = submodules["layers1.0.attention.output.dropout"].p
        kwargs["intermediate_size1"] = submodules["layers1.0.intermediate.dense"].out_features
        kwargs["activation"] = submodules["layers1.0.intermediate"].intermediate_act_fn

        return kwargs

    def _load_from_pretrained_module(
        self,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Loads the weights of the `pretrained_module` into the instance.
        Optionally, a `mapping` is specified for any differences in parameter names
        between `pretrained_module` and the instance.
        """
        # FIX: ignore_absent_parameters should be a general option.
        if mapping is None:
            mapping = self._construct_default_mapping(source)
            # mapping = self._default_mapping
        pretrained_parameters = dict(pretrained_module.named_parameters())
        ignore_absent_parameters = ["layers2", "c_layer"]  # FIX: specific to source.
        for name, parameter in self.named_parameters():
            pretrained_name = name
            for key, val in mapping.items():
                # so that we replace the names of submodules too.
                # eg. module.key.anothermodule --> module.val.anothermodule
                pretrained_name = pretrained_name.replace(key, val)

            if not any(
                [pretrained_name.startswith(paraname) for paraname in ignore_absent_parameters]
            ):
                if pretrained_name not in pretrained_parameters:
                    raise ValueError(
                        f"Couldn't find a matching parameter for {name}. Is this module "
                        "compatible with the pretrained module you're using?"
                    )
                parameter.data.copy_(pretrained_parameters[pretrained_name].data)

    @classmethod
    def from_pretrained_module(  # type: ignore
        cls,
        pretrained_module: torch.nn.Module,
        num_hidden_layers2: int,
        hidden_size2: int,
        combined_hidden_size: int,
        intermediate_size2: int,
        attention_dropout2: float,
        hidden_dropout2: float,
        biattention_id1: List[int],
        biattention_id2: List[int],
        fixed_layer1: int,
        fixed_layer2: int,
        fast_mode: bool = False,
        with_coattention: bool = True,
        in_batch_pairs: bool = False,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        # **kwargs,
    ):
        """
        The `pretrained_module` only supplies one of the modalities.
        """
        final_kwargs = {}
        final_kwargs.update(cls._get_input_arguments(pretrained_module, source, mapping))
        final_kwargs["num_hidden_layers2"] = num_hidden_layers2
        final_kwargs["hidden_size2"] = hidden_size2
        final_kwargs["combined_hidden_size"] = combined_hidden_size
        final_kwargs["intermediate_size2"] = intermediate_size2
        final_kwargs["attention_dropout2"] = attention_dropout2
        final_kwargs["hidden_dropout2"] = hidden_dropout2
        final_kwargs["biattention_id1"] = biattention_id1
        final_kwargs["biattention_id2"] = biattention_id2
        final_kwargs["fixed_layer1"] = fixed_layer1
        final_kwargs["fixed_layer2"] = fixed_layer2
        final_kwargs["fast_mode"] = fast_mode
        final_kwargs["with_coattention"] = with_coattention
        final_kwargs["in_batch_pairs"] = in_batch_pairs

        return super().from_pretrained_module(pretrained_module, source, mapping, **final_kwargs)
