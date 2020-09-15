import copy

from allennlp.common import Params
from allennlp.common.testing import assert_equal_parameters
from allennlp.modules.transformer import TransformerEncoder
from allennlp.common.testing import AllenNlpTestCase

from transformers.modeling_auto import AutoModel


class TestTransformerEncoder(AllenNlpTestCase):
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

        self.transformer_encoder = TransformerEncoder.from_params(params)

        self.pretrained = AutoModel.from_pretrained("bert-base-uncased")

    def test_can_construct_from_params(self):

        modules = dict(self.transformer_encoder.named_modules())
        assert len(modules["layers"]) == self.params_dict["num_hidden_layers"]

    # def test_forward_runs(self):
    #     self.transformer_encoder.forward(torch.randn(2, 3, 6), torch.randn(2, 2, 3, 3))

    def test_loading_from_pretrained_weights(self):
        pretrained_module = self.pretrained.encoder
        module = TransformerEncoder.from_pretrained_module(pretrained_module)
        assert_equal_parameters(pretrained_module, module)

    def test_loading_partial_pretrained_weights(self):

        args = list(TransformerEncoder._get_input_arguments(self.pretrained.encoder))
        # The pretrained module has 12 bert layers, while the instance will have only 3.
        args[0] = 3
        transformer_encoder = TransformerEncoder(*args)
        transformer_encoder._load_from_pretrained_module(self.pretrained.encoder)
        assert_equal_parameters(self.pretrained.encoder, transformer_encoder)
