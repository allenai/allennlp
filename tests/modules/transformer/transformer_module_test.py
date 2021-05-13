import torch
from torch.nn import Parameter

from allennlp.common.testing import assert_equal_parameters, assert_allclose
from allennlp.modules.transformer import TransformerModule
from allennlp.common.testing import AllenNlpTestCase


class TestTransformerModule(AllenNlpTestCase):
    def test_get_mapped_state_dict(self):
        class InternalOld(torch.nn.Module):
            def __init__(self, inp, out):
                super().__init__()
                self.ff = torch.nn.Linear(inp, out)
                self.p = Parameter(torch.randn(out, out))
                self.register_buffer("b", torch.randn(inp, inp))

            def forward(self, x):
                x = self.ff(x).matmul(self.p)
                return x

        class InternalNew(TransformerModule):
            _pretrained_mapping = {"ff": "linear", "p": "param", "b": "buffer"}

            def __init__(self, inp, out):
                super().__init__()
                self.linear = torch.nn.Linear(inp, out)
                self.param = Parameter(torch.randn(out, out))
                self.register_buffer("buffer", torch.randn(inp, inp))

            def forward(self, x):
                x = self.linear(x).matmul(self.param)
                return x

        class ExternalOld(torch.nn.Module):
            def __init__(self, inp, out):
                super().__init__()
                self.internal = InternalOld(inp, out)
                self.p = Parameter(torch.randn(out, out))

            def forward(self, x):
                x = self.internal(x).matmul(self.p)
                return x

        class ExternalNew(TransformerModule):
            _pretrained_mapping = {"internal": "internal_layer", "p": "param"}

            def __init__(self, inp, out):
                super().__init__()
                self.internal_layer = InternalNew(inp, out)
                self.param = Parameter(torch.randn(out, out))

            def forward(self, x):
                x = self.internal_layer(x).matmul(self.param)
                return x

        eold = ExternalOld(3, 5)
        state_dict_old = eold.state_dict()

        enew = ExternalNew(3, 5)
        state_dict_new = enew._get_mapped_state_dict(state_dict_old)
        assert set(state_dict_new.keys()) == set(
            [
                "internal_layer.linear.weight",
                "internal_layer.linear.bias",
                "internal_layer.param",
                "internal_layer.buffer",
                "param",
            ]
        )

        enew.load_state_dict(state_dict_new)

        x = torch.randn(4, 3)
        out_old = eold(x)
        out_new = enew(x)
        assert_allclose(out_old, out_new)

        assert_equal_parameters(
            eold,
            enew,
            mapping={
                "internal_layer.linear.weight": "internal.ff.weight",
                "internal_layer.linear.bias": "internal.ff.bias",
                "internal_layer.param": "internal.p",
                "internal_layer.buffer": "internal.b",
                "param": "p",
            },
        )
