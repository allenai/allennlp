import copy

from allennlp.common import Params
from allennlp.common.testing import assert_equal_parameters
from allennlp.modules.transformer import BiModalEncoder
from allennlp.common.testing import AllenNlpTestCase

from transformers.modeling_auto import AutoModel


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
            "num_attention_heads": 2,
            "attention_dropout1": 0.1,
            "hidden_dropout1": 0.2,
            "attention_dropout2": 0.1,
            "hidden_dropout2": 0.2,
            "activation": "relu",
            "biattention_id1": [1, 2],
            "biattention_id2": [1, 2],
            "fixed_layer1": 2,
            "fixed_layer2": 2,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.bimodal_encoder = BiModalEncoder.from_params(params)

        self.pretrained = AutoModel.from_pretrained("bert-base-uncased")

    def test_can_construct_from_params(self):

        modules = dict(self.bimodal_encoder.named_modules())
        assert len(modules["layers1"]) == self.params_dict["num_hidden_layers1"]
        assert len(modules["layers2"]) == self.params_dict["num_hidden_layers2"]

    def test_forward_runs(self):
        pass

    def test_loading_from_pretrained_weights(self):
        pretrained_module = self.pretrained.encoder
        required_kwargs = [
            "num_hidden_layers2",
            "hidden_size2",
            "combined_hidden_size",
            "intermediate_size2",
            "attention_dropout2",
            "hidden_dropout2",
            "biattention_id1",
            "biattention_id2",
            "fixed_layer1",
            "fixed_layer2",
        ]
        kwargs = {key: self.params_dict[key] for key in required_kwargs}
        module = BiModalEncoder.from_pretrained_module(pretrained_module, **kwargs)
        assert_equal_parameters(pretrained_module, module, ignore_missing=True)
