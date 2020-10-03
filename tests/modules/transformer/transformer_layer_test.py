import copy
import torch

from allennlp.common import Params
from allennlp.common import cached_transformers
from allennlp.common.testing import assert_equal_parameters
from allennlp.modules.transformer import AttentionLayer, TransformerLayer
from allennlp.common.testing import AllenNlpTestCase


class TestAttentionLayer(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 6,
            "num_attention_heads": 2,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.2,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.attention_layer = AttentionLayer.from_params(params)

    def test_can_construct_from_params(self):

        attention_layer = self.attention_layer

        assert attention_layer.self.num_attention_heads == self.params_dict["num_attention_heads"]
        assert attention_layer.self.attention_head_size == int(
            self.params_dict["hidden_size"] / self.params_dict["num_attention_heads"]
        )
        assert (
            attention_layer.self.all_head_size
            == self.params_dict["num_attention_heads"] * attention_layer.self.attention_head_size
        )
        assert attention_layer.self.query.in_features == self.params_dict["hidden_size"]
        assert attention_layer.self.key.in_features == self.params_dict["hidden_size"]
        assert attention_layer.self.value.in_features == self.params_dict["hidden_size"]
        assert attention_layer.self.dropout.p == self.params_dict["attention_dropout"]

        assert attention_layer.output.dense.in_features == self.params_dict["hidden_size"]
        assert attention_layer.output.dense.out_features == self.params_dict["hidden_size"]
        assert (
            attention_layer.output.layer_norm.normalized_shape[0] == self.params_dict["hidden_size"]
        )
        assert attention_layer.output.dropout.p == self.params_dict["hidden_dropout"]

    def test_forward_runs(self):

        self.attention_layer.forward(torch.randn(2, 3, 6), torch.randn(2, 2, 3, 3))


class TestTransformerLayer(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "hidden_size": 6,
            "intermediate_size": 3,
            "num_attention_heads": 2,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.2,
            "activation": "relu",
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.transformer_layer = TransformerLayer.from_params(params)
        self.pretrained_name = "bert-base-uncased"

        self.pretrained = cached_transformers.get(self.pretrained_name, False)

    def test_can_construct_from_params(self):

        transformer_layer = self.transformer_layer

        assert (
            transformer_layer.attention.self.num_attention_heads
            == self.params_dict["num_attention_heads"]
        )
        assert transformer_layer.attention.self.attention_head_size == int(
            self.params_dict["hidden_size"] / self.params_dict["num_attention_heads"]
        )
        assert (
            transformer_layer.attention.self.all_head_size
            == self.params_dict["num_attention_heads"]
            * transformer_layer.attention.self.attention_head_size
        )
        assert transformer_layer.attention.self.query.in_features == self.params_dict["hidden_size"]
        assert transformer_layer.attention.self.key.in_features == self.params_dict["hidden_size"]
        assert transformer_layer.attention.self.value.in_features == self.params_dict["hidden_size"]
        assert transformer_layer.attention.self.dropout.p == self.params_dict["attention_dropout"]

        assert (
            transformer_layer.attention.output.dense.in_features == self.params_dict["hidden_size"]
        )
        assert (
            transformer_layer.attention.output.dense.out_features == self.params_dict["hidden_size"]
        )
        assert (
            transformer_layer.attention.output.layer_norm.normalized_shape[0]
            == self.params_dict["hidden_size"]
        )
        assert transformer_layer.attention.output.dropout.p == self.params_dict["hidden_dropout"]

        assert transformer_layer.intermediate.dense.in_features == self.params_dict["hidden_size"]
        assert (
            transformer_layer.intermediate.dense.out_features
            == self.params_dict["intermediate_size"]
        )

        assert transformer_layer.output.dense.in_features == self.params_dict["intermediate_size"]
        assert transformer_layer.output.dense.out_features == self.params_dict["hidden_size"]

        assert (
            transformer_layer.output.layer_norm.normalized_shape[0]
            == self.params_dict["hidden_size"]
        )

        assert transformer_layer.output.dropout.p == self.params_dict["hidden_dropout"]

    def test_forward_runs(self):

        self.transformer_layer.forward(torch.randn(2, 3, 6), torch.randn(2, 2, 3, 3))

    def test_loading_from_pretrained_weights(self):

        # Hacky way to get a bert layer.
        for i, pretrained_module in enumerate(self.pretrained.encoder.layer.modules()):
            if i == 1:
                break

        module = TransformerLayer.from_pretrained_module(pretrained_module)
        mapping = {
            val: key for key, val in module._construct_default_mapping("huggingface").items()
        }
        assert_equal_parameters(pretrained_module, module, mapping=mapping)

    def test_loading_from_pretrained_weights_using_model_name(self):

        # Hacky way to get a bert layer.
        for i, pretrained_module in enumerate(self.pretrained.encoder.layer.modules()):
            if i == 1:
                break

        module = TransformerLayer.from_pretrained_module(self.pretrained_name)
        mapping = {
            val: key for key, val in module._construct_default_mapping("huggingface").items()
        }
        assert_equal_parameters(pretrained_module, module, mapping=mapping)
