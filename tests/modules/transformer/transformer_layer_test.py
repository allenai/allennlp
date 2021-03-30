import copy
import torch
import pytest

from allennlp.common import Params
from allennlp.common import cached_transformers
from allennlp.common.testing import assert_equal_parameters
from allennlp.modules.transformer import AttentionLayer, TransformerLayer
from allennlp.common.testing import AllenNlpTestCase

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertLayer
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaLayer
from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraAttention, ElectraLayer

ATTENTION_PARAMS_DICT = {
    "hidden_size": 6,
    "num_attention_heads": 2,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.2,
}


def get_attention_modules(params_dict):
    modules = {}
    params = copy.deepcopy(params_dict)
    params["attention_probs_dropout_prob"] = params.pop("attention_dropout")
    params["hidden_dropout_prob"] = params.pop("hidden_dropout")

    torch.manual_seed(1234)
    hf_module = BertAttention(BertConfig(**params))
    modules["bert"] = hf_module

    torch.manual_seed(1234)
    hf_module = RobertaAttention(RobertaConfig(**params))
    modules["roberta"] = hf_module

    torch.manual_seed(1234)
    hf_module = ElectraAttention(ElectraConfig(**params))
    modules["electra"] = hf_module

    return modules


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
        attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])
        self.attention_layer.forward(torch.randn(2, 3, 6), attention_mask=attention_mask)

    @pytest.mark.parametrize(
        "module_name, hf_module", get_attention_modules(ATTENTION_PARAMS_DICT).items()
    )
    def test_forward_against_huggingface_outputs(self, module_name, hf_module):
        hidden_states = torch.randn(2, 3, 6)
        attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])

        attention = AttentionLayer.from_pretrained_module(hf_module)

        torch.manual_seed(1234)
        output = attention.forward(hidden_states, attention_mask=attention_mask)
        # We do this because bert, roberta, electra process the attention_mask at the model level.
        attention_mask_hf = (attention_mask == 0).view((2, 1, 1, 3)).expand(2, 2, 3, 3) * -10e5
        torch.manual_seed(1234)
        hf_output = hf_module.forward(hidden_states, attention_mask=attention_mask_hf)

        assert torch.allclose(output[0], hf_output[0])

    @pytest.mark.parametrize(
        "pretrained_name",
        [
            "bert-base-uncased",
            "roberta-base",
        ],
    )
    def test_loading_from_pretrained_weights_using_model_name(self, pretrained_name):

        torch.manual_seed(1234)
        pretrained = cached_transformers.get(pretrained_name, False)

        if "distilbert" in pretrained_name:
            encoder = pretrained.transformer
        else:
            encoder = pretrained.encoder
        # Hacky way to get a bert layer.
        for i, pretrained_module in enumerate(encoder.layer.modules()):
            if i == 1:
                break

        pretrained_module = pretrained_module.attention

        torch.manual_seed(1234)
        module = AttentionLayer.from_pretrained_module(pretrained_name)
        mapping = {
            val: key
            for key, val in module._construct_default_mapping(
                pretrained_module, "huggingface", {}
            ).items()
        }
        assert_equal_parameters(pretrained_module, module, mapping=mapping)

        batch_size = 2
        seq_len = 768
        dim = module.self.query.in_features
        hidden_states = torch.randn(batch_size, seq_len, dim)
        attention_mask = torch.randint(0, 2, (batch_size, seq_len))
        mask_reshp = (batch_size, 1, 1, dim)
        attention_mask_hf = (attention_mask == 0).view(mask_reshp).expand(
            batch_size, 12, seq_len, seq_len
        ) * -10e5

        # setting to eval mode to avoid non-deterministic dropout.
        module = module.eval()
        pretrained_module = pretrained_module.eval()

        torch.manual_seed(1234)
        output = module.forward(hidden_states, attention_mask=attention_mask.squeeze())[0]
        torch.manual_seed(1234)
        hf_output = pretrained_module.forward(hidden_states, attention_mask=attention_mask_hf)[0]

        assert torch.allclose(output, hf_output, atol=1e-04)


LAYER_PARAMS_DICT = {
    "hidden_size": 6,
    "intermediate_size": 3,
    "num_attention_heads": 2,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.2,
    "activation": "relu",
}


def get_layer_modules(params_dict):
    modules = {}
    params = copy.deepcopy(params_dict)
    params["attention_probs_dropout_prob"] = params.pop("attention_dropout")
    params["hidden_dropout_prob"] = params.pop("hidden_dropout")

    # bert, roberta, electra, layoutlm self attentions have the same code.

    torch.manual_seed(1234)
    hf_module = BertLayer(BertConfig(**params))
    modules["bert"] = hf_module

    torch.manual_seed(1234)
    hf_module = RobertaLayer(RobertaConfig(**params))
    modules["roberta"] = hf_module

    torch.manual_seed(1234)
    hf_module = ElectraLayer(ElectraConfig(**params))
    modules["electra"] = hf_module

    return modules


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
        attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])
        self.transformer_layer.forward(torch.randn(2, 3, 6), attention_mask=attention_mask)

        with pytest.raises(AssertionError):
            self.transformer_layer.forward(
                torch.randn(2, 3, 6),
                attention_mask=attention_mask,
                encoder_hidden_states=torch.randn(2, 3, 6),
            )

    def test_cross_attention(self):
        params = copy.deepcopy(self.params_dict)
        params["add_cross_attention"] = True

        params = Params(params)

        transformer_layer = TransformerLayer.from_params(params)
        assert hasattr(transformer_layer, "cross_attention")

        attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])
        transformer_layer.forward(
            torch.randn(2, 3, 6),
            attention_mask=attention_mask,
            encoder_hidden_states=torch.randn(2, 3, 6),
        )

        transformer_layer_new = TransformerLayer.from_pretrained_module(
            transformer_layer, source="allennlp"
        )

        assert hasattr(transformer_layer_new, "cross_attention")

    def test_loading_from_pretrained_weights(self):

        # Hacky way to get a bert layer.
        for i, pretrained_module in enumerate(self.pretrained.encoder.layer.modules()):
            if i == 1:
                break

        module = TransformerLayer.from_pretrained_module(pretrained_module)
        mapping = {
            val: key
            for key, val in module._construct_default_mapping(
                pretrained_module, "huggingface", {}
            ).items()
        }
        assert_equal_parameters(pretrained_module, module, mapping=mapping)

    @pytest.mark.parametrize("module_name, hf_module", get_layer_modules(LAYER_PARAMS_DICT).items())
    def test_forward_against_huggingface_outputs(self, module_name, hf_module):
        hidden_states = torch.randn(2, 3, 6)
        attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])

        layer = TransformerLayer.from_pretrained_module(hf_module)

        torch.manual_seed(1234)
        output = layer.forward(hidden_states, attention_mask=attention_mask)
        # We do this because bert, roberta, electra process the attention_mask at the model level.
        attention_mask_hf = (attention_mask == 0).view((2, 1, 1, 3)).expand(2, 2, 3, 3) * -10e5
        torch.manual_seed(1234)
        hf_output = hf_module.forward(hidden_states, attention_mask=attention_mask_hf)

        assert torch.allclose(output[0], hf_output[0])

    @pytest.mark.parametrize(
        "pretrained_name",
        [
            "bert-base-uncased",
            "roberta-base",
        ],
    )
    def test_loading_from_pretrained_weights_using_model_name(self, pretrained_name):

        torch.manual_seed(1234)
        pretrained = cached_transformers.get(pretrained_name, False)

        if "distilbert" in pretrained_name:
            encoder = pretrained.transformer
        else:
            encoder = pretrained.encoder
        # Hacky way to get a bert layer.
        for i, pretrained_module in enumerate(encoder.layer.modules()):
            if i == 1:
                break

        pretrained_module = pretrained_module

        torch.manual_seed(1234)
        module = TransformerLayer.from_pretrained_module(pretrained_name)
        mapping = {
            val: key
            for key, val in module._construct_default_mapping(
                pretrained_module, "huggingface", {}
            ).items()
        }
        assert_equal_parameters(pretrained_module, module, mapping=mapping)

        batch_size = 2
        seq_len = 768
        dim = module.attention.self.query.in_features
        hidden_states = torch.randn(batch_size, seq_len, dim)
        attention_mask = torch.randint(0, 2, (batch_size, seq_len))
        mask_reshp = (batch_size, 1, 1, dim)
        attention_mask_hf = (attention_mask == 0).view(mask_reshp).expand(
            batch_size, 12, seq_len, seq_len
        ) * -10e5

        # setting to eval mode to avoid non-deterministic dropout.
        module = module.eval()
        pretrained_module = pretrained_module.eval()

        torch.manual_seed(1234)
        output = module.forward(hidden_states, attention_mask=attention_mask.squeeze())[0]
        torch.manual_seed(1234)
        hf_output = pretrained_module.forward(hidden_states, attention_mask=attention_mask_hf)[0]

        assert torch.allclose(output, hf_output, atol=1e-04)
