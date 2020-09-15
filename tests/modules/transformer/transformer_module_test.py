import torch

from allennlp.common.testing import assert_equal_parameters
from allennlp.modules.transformer import TransformerModule
from allennlp.common.testing import AllenNlpTestCase


class TestTransformerModule(AllenNlpTestCase):
    def test_can_load_pretrained_weights(self):
        class InternalOld(torch.nn.Module):
            def __init__(self, inp, out):
                super().__init__()
                self.ff = torch.nn.Linear(inp, out)

            def forward(self, x):
                x = self.ff(x)
                return x

        class InternalNew(TransformerModule):
            def __init__(self, inp, out):
                super().__init__()
                self.linear = torch.nn.Linear(inp, out)

            def _construct_default_mapping(self, source):
                self._default_mapping = {"linear": "ff"}

            def forward(self, x):
                x = self.linear(x)
                return x

        class ExternalOld(torch.nn.Module):
            def __init__(self, inp, out):
                super().__init__()
                self.internal = InternalOld(inp, out)

            def forward(self, x):
                x = self.internal(x)
                return x

        class External(TransformerModule):
            _huggingface_mapping = {"internal_layer": "internal"}

            def __init__(self, inp, out):
                super().__init__()
                self.internal_layer = InternalNew(inp, out)

            def forward(self, x):
                x = self.internal_layer(x)
                return x

        iold = InternalOld(3, 5)
        x = torch.randn(4, 3)
        iold.forward(x)
        inew = InternalNew(3, 5)
        inew._load_from_pretrained_module(iold)

        assert_equal_parameters(iold, inew)

        eold = ExternalOld(3, 5)
        x = torch.randn(4, 3)
        eold.forward(x)

        enew = External(3, 5)
        enew._load_from_pretrained_module(eold)

        assert_equal_parameters(eold, enew)
