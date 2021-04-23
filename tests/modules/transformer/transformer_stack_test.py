import copy
import torch
import pytest

from allennlp.common import Params
from allennlp.common import cached_transformers

from allennlp.common.testing import assert_equal_parameters
from allennlp.modules.transformer import TransformerStack, TransformerLayer
from allennlp.common.testing import AllenNlpTestCase

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaEncoder
from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraEncoder

PARAMS_DICT = {
    "num_hidden_layers": 3,
    "hidden_size": 6,
    "intermediate_size": 3,
    "num_attention_heads": 2,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.2,
    "activation": "relu",
}


def get_modules(params_dict):
    modules = {}
    params = copy.deepcopy(params_dict)
    params["attention_probs_dropout_prob"] = params.pop("attention_dropout")
    params["hidden_dropout_prob"] = params.pop("hidden_dropout")

    torch.manual_seed(1234)
    hf_module = BertEncoder(BertConfig(**params))
    modules["bert"] = hf_module

    torch.manual_seed(1234)
    hf_module = RobertaEncoder(RobertaConfig(**params))
    modules["roberta"] = hf_module

    torch.manual_seed(1234)
    hf_module = ElectraEncoder(ElectraConfig(**params))
    modules["electra"] = hf_module

    return modules


class TestTransformerStack(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "num_hidden_layers": 3,
            "hidden_size": 6,
            "intermediate_size": 3,
            "num_attention_heads": 2,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.2,
            "activation": "relu",
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.transformer_stack = TransformerStack.from_params(params)

        self.pretrained_name = "bert-base-uncased"

        self.pretrained = cached_transformers.get(self.pretrained_name, False)

    def test_can_construct_from_params(self):

        modules = dict(self.transformer_stack.named_modules())
        assert len(modules["layers"]) == self.params_dict["num_hidden_layers"]

    def test_forward_runs(self):
        self.transformer_stack.forward(torch.randn(2, 3, 6), attention_mask=torch.randn(2, 3))

        with pytest.raises(AssertionError):
            self.transformer_stack.forward(
                torch.randn(2, 3, 6),
                attention_mask=torch.randn(2, 3),
                encoder_hidden_states=torch.randn(2, 3, 6),
            )

    def test_layer_same_as_params(self):
        params = copy.deepcopy(self.params_dict)
        num_hidden_layers = params.pop("num_hidden_layers")
        # params = Params(params)

        torch.manual_seed(1234)
        transformer_layer = TransformerLayer(**params)
        transformer_stack_from_layer = TransformerStack(num_hidden_layers, transformer_layer)
        torch.manual_seed(1234)
        transformer_stack_from_params = TransformerStack(num_hidden_layers, **params)

        hidden_states = torch.randn(2, 3, 6)
        attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])

        transformer_stack_from_layer.eval()
        transformer_stack_from_params.eval()

        torch.manual_seed(1234)
        layer_output = transformer_stack_from_layer.forward(
            hidden_states, attention_mask=attention_mask
        )

        torch.manual_seed(1234)
        params_output = transformer_stack_from_params.forward(
            hidden_states, attention_mask=attention_mask
        )

        assert torch.allclose(layer_output[0], params_output[0])

    def test_cross_attention(self):
        params = copy.deepcopy(self.params_dict)
        params["add_cross_attention"] = True

        params = Params(params)

        transformer_stack = TransformerStack.from_params(params)
        modules = dict(transformer_stack.named_modules())

        assert hasattr(modules["layers.0"], "cross_attention")

        attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])
        transformer_stack.forward(
            torch.randn(2, 3, 6),
            attention_mask=attention_mask,
            encoder_hidden_states=torch.randn(2, 3, 6),
        )

        transformer_stack_new = TransformerStack.from_pretrained_module(
            transformer_stack, source="allennlp"
        )

        new_modules = dict(transformer_stack_new.named_modules())
        assert hasattr(new_modules["layers.0"], "cross_attention")

    def test_loading_from_pretrained_weights(self):
        pretrained_module = self.pretrained.encoder
        module = TransformerStack.from_pretrained_module(pretrained_module)
        mapping = {
            val: key
            for key, val in module._construct_default_mapping(
                pretrained_module, "huggingface", {}
            ).items()
        }
        assert_equal_parameters(pretrained_module, module, mapping)

    def test_loading_partial_pretrained_weights(self):

        kwargs = TransformerStack._get_input_arguments(self.pretrained.encoder)
        # The pretrained module has 12 bert layers, while the instance will have only 3.
        kwargs["num_hidden_layers"] = 3
        transformer_stack = TransformerStack(**kwargs)
        transformer_stack._load_from_pretrained_module(self.pretrained.encoder)
        mapping = {
            val: key
            for key, val in transformer_stack._construct_default_mapping(
                self.pretrained.encoder, "huggingface", {}
            ).items()
        }
        assert_equal_parameters(
            self.pretrained.encoder,
            transformer_stack,
            mapping,
        )

    @pytest.mark.skip("Takes up too much memory")
    @pytest.mark.parametrize("module_name, hf_module", get_modules(PARAMS_DICT).items())
    def test_forward_against_huggingface_outputs(self, module_name, hf_module):
        hidden_states = torch.randn(2, 3, 6)
        attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])

        stack = TransformerStack.from_pretrained_module(hf_module)

        torch.manual_seed(1234)
        output = stack.forward(hidden_states, attention_mask=attention_mask)
        # We do this because bert, roberta, electra process the attention_mask at the model level.
        attention_mask_hf = (attention_mask == 0).view((2, 1, 1, 3)).expand(2, 2, 3, 3) * -10e5
        torch.manual_seed(1234)
        hf_output = hf_module.forward(hidden_states, attention_mask=attention_mask_hf)

        assert torch.allclose(output[0], hf_output[0])

    @pytest.mark.parametrize(
        "pretrained_name",
        [
            "bert-base-uncased",
        ],
    )
    def test_loading_from_pretrained_weights_using_model_name(self, pretrained_name):

        torch.manual_seed(1234)
        pretrained = cached_transformers.get(pretrained_name, False)

        if "distilbert" in pretrained_name:
            pretrained_module = pretrained.transformer
        else:
            pretrained_module = pretrained.encoder

        torch.manual_seed(1234)
        module = TransformerStack.from_pretrained_module(pretrained_name)
        mapping = {
            val: key
            for key, val in module._construct_default_mapping(
                pretrained_module, "huggingface", {}
            ).items()
        }
        assert_equal_parameters(pretrained_module, module, mapping=mapping)

        batch_size = 1
        seq_len = 768
        dim = dict(module.named_modules())["layers.0.attention.self.query"].in_features
        hidden_states = torch.randn(batch_size, seq_len, dim)
        attention_mask = torch.randint(0, 2, (batch_size, seq_len))
        mask_reshp = (batch_size, 1, 1, dim)
        attention_mask_hf = (attention_mask == 0).view(mask_reshp)
        attention_mask_hf = attention_mask_hf.expand(batch_size, 12, seq_len, seq_len) * -10e5

        # setting to eval mode to avoid non-deterministic dropout.
        module = module.eval()
        pretrained_module = pretrained_module.eval()

        torch.manual_seed(1234)
        output = module.forward(hidden_states, attention_mask=attention_mask.squeeze())[0]
        torch.manual_seed(1234)
        hf_output = pretrained_module.forward(hidden_states, attention_mask=attention_mask_hf)[0]

        assert torch.allclose(output, hf_output)
