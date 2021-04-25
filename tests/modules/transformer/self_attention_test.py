import copy
import torch
import pytest

from allennlp.common import Params
from allennlp.common import cached_transformers
from allennlp.common.testing import assert_equal_parameters, AllenNlpTestCase
from allennlp.modules.transformer import SelfAttention
from allennlp.nn.util import min_value_of_dtype

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraSelfAttention
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention

PARAMS_DICT = {
    "hidden_size": 6,
    "num_attention_heads": 2,
    "dropout": 0.0,
}


def get_modules(params_dict):
    modules = {}
    params = copy.deepcopy(params_dict)
    params["attention_probs_dropout_prob"] = params.pop("dropout")

    # bert, roberta, electra self attentions have the same code.

    torch.manual_seed(1234)
    hf_module = BertSelfAttention(BertConfig(**params))
    modules["bert"] = hf_module

    torch.manual_seed(1234)
    hf_module = RobertaSelfAttention(RobertaConfig(**params))
    modules["roberta"] = hf_module

    torch.manual_seed(1234)
    hf_module = ElectraSelfAttention(ElectraConfig(**params))
    modules["electra"] = hf_module

    torch.manual_seed(1234)
    distilparams = copy.deepcopy(params_dict)
    distilparams["n_heads"] = distilparams.pop("num_attention_heads")
    distilparams["dim"] = distilparams.pop("hidden_size")
    distilparams["attention_dropout"] = distilparams.pop("dropout")
    hf_module = MultiHeadSelfAttention(DistilBertConfig(**distilparams))
    modules["distilbert"] = hf_module

    return modules


class TestSelfAttention(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {key: val for key, val in PARAMS_DICT.items()}

        params = Params(copy.deepcopy(self.params_dict))

        self.self_attention = SelfAttention.from_params(params)

    def test_can_construct_from_params(self):
        assert self.self_attention.num_attention_heads == self.params_dict["num_attention_heads"]
        assert self.self_attention.attention_head_size == int(
            self.params_dict["hidden_size"] / self.params_dict["num_attention_heads"]
        )

        assert (
            self.self_attention.all_head_size
            == self.params_dict["num_attention_heads"] * self.self_attention.attention_head_size
        )

        assert self.self_attention.query.in_features == self.params_dict["hidden_size"]
        assert self.self_attention.key.in_features == self.params_dict["hidden_size"]
        assert self.self_attention.value.in_features == self.params_dict["hidden_size"]

        assert self.self_attention.dropout.p == self.params_dict["dropout"]

    @pytest.mark.skip("Takes up too much memory")
    @pytest.mark.parametrize("module_name, hf_module", get_modules(PARAMS_DICT).items())
    def test_forward_against_huggingface_output(self, module_name, hf_module):
        hidden_states = torch.randn(2, 3, 6)
        attention_mask = torch.tensor([[0, 1, 0], [1, 1, 0]])

        torch.manual_seed(1234)
        self_attention = SelfAttention.from_pretrained_module(hf_module)

        output = self_attention.forward(hidden_states, attention_mask=attention_mask)
        if module_name == "distilbert":
            hf_output = hf_module.forward(
                hidden_states, hidden_states, hidden_states, mask=attention_mask
            )
        else:
            # We do this because bert, roberta, electra process the attention_mask at the model level.
            attention_mask_hf = (attention_mask == 0).view((2, 1, 1, 3)).expand(2, 2, 3, 3) * -10e5
            hf_output = hf_module.forward(hidden_states, attention_mask=attention_mask_hf)

        assert torch.allclose(output[0], hf_output[0])

    @pytest.mark.skip("Takes up too much memory")
    @pytest.mark.parametrize(
        "pretrained_name",
        [
            "bert-base-uncased",
            "roberta-base",
            "google/electra-base-generator",
            "distilbert-base-uncased",
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

        # Get the self attention layer.
        if "distilbert" in pretrained_name:
            pretrained_module = pretrained_module.attention
        else:
            pretrained_module = pretrained_module.attention.self

        torch.manual_seed(1234)
        module = SelfAttention.from_pretrained_module(pretrained_name)
        mapping = {
            val: key
            for key, val in module._construct_default_mapping(
                pretrained_module, "huggingface", {}
            ).items()
        }
        assert_equal_parameters(pretrained_module, module, mapping=mapping)

        batch_size = 2
        seq_len = 3
        dim = module.query.in_features
        hidden_states = torch.randn(batch_size, seq_len, dim)
        attention_mask = torch.randint(0, 2, (batch_size, 1, 1, seq_len))

        # setting to eval mode to avoid non-deterministic dropout.
        module = module.eval()
        pretrained_module = pretrained_module.eval()

        torch.manual_seed(1234)
        output = module.forward(hidden_states, attention_mask=attention_mask.squeeze())[0]
        if "distilbert" in pretrained_name:
            torch.manual_seed(1234)
            hf_output = pretrained_module.forward(
                hidden_states, hidden_states, hidden_states, mask=attention_mask
            )[0]
        else:
            # The attn_mask is processed outside the self attention module in HF bert models.
            attention_mask = (~(attention_mask == 1)) * min_value_of_dtype(hidden_states.dtype)
            torch.manual_seed(1234)
            hf_output = pretrained_module.forward(hidden_states, attention_mask=attention_mask)[0]

        assert torch.allclose(output, hf_output)
