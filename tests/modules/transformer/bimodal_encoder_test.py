import copy
import torch
from allennlp.common import Params
from allennlp.common import cached_transformers
from allennlp.common.testing import assert_equal_parameters
from allennlp.modules.transformer import BiModalEncoder
from allennlp.common.testing import AllenNlpTestCase


class TestBiModalEncoder(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "num_hidden_layers1": 3,
            "num_hidden_layers2": 3,
            "hidden_size1": 12,
            "hidden_size2": 12,
            "combined_hidden_size": 12,
            "intermediate_size1": 3,
            "intermediate_size2": 3,
            "num_attention_heads1": 4,
            "num_attention_heads2": 6,
            "combined_num_attention_heads": 2,
            "attention_dropout1": 0.1,
            "hidden_dropout1": 0.2,
            "attention_dropout2": 0.1,
            "hidden_dropout2": 0.2,
            "activation": "relu",
            "biattention_id1": [1, 2],
            "biattention_id2": [1, 2],
            "fixed_layer1": 1,
            "fixed_layer2": 1,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.bimodal_encoder = BiModalEncoder.from_params(params)

        self.pretrained = cached_transformers.get("bert-base-uncased", False)

    def test_can_construct_from_params(self):

        modules = dict(self.bimodal_encoder.named_modules())
        assert len(modules["layers1"]) == self.params_dict["num_hidden_layers1"]
        assert len(modules["layers2"]) == self.params_dict["num_hidden_layers2"]

    def test_forward_runs(self):

        embedding1 = torch.randn(16, 34, self.params_dict["hidden_size1"])
        embedding2 = torch.randn(16, 2, self.params_dict["hidden_size2"])
        attn_mask1 = torch.randint(0, 2, (16, 1, 1, 34)) == 1
        attn_mask2 = torch.randint(0, 2, (16, 1, 1, 2)) == 1

        self.bimodal_encoder.forward(embedding1, embedding2, attn_mask1, attn_mask2)

    def test_loading_from_pretrained_weights(self):
        pretrained_module = self.pretrained.encoder
        required_kwargs = [
            "num_hidden_layers2",
            "hidden_size2",
            "combined_hidden_size",
            "intermediate_size2",
            "num_attention_heads2",
            "combined_num_attention_heads",
            "attention_dropout2",
            "hidden_dropout2",
            "biattention_id1",
            "biattention_id2",
            "fixed_layer1",
            "fixed_layer2",
        ]
        kwargs = {key: self.params_dict[key] for key in required_kwargs}
        module = BiModalEncoder.from_pretrained_module(pretrained_module, **kwargs)
        mapping = {
            val: key
            for key, val in module._construct_default_mapping(
                pretrained_module, "huggingface", {}
            ).items()
        }
        assert_equal_parameters(
            pretrained_module,
            module,
            ignore_missing=True,
            mapping=mapping,
        )

    def test_default_parameters(self):
        encoder = BiModalEncoder()
        embedding1 = torch.randn(16, 34, 1024)
        embedding2 = torch.randn(16, 2, 1024)
        attn_mask1 = torch.randint(0, 2, (16, 1, 1, 34)) == 1
        attn_mask2 = torch.randint(0, 2, (16, 1, 1, 2)) == 1

        encoder.forward(embedding1, embedding2, attn_mask1, attn_mask2)
