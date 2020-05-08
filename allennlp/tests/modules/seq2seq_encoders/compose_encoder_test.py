import torch
import numpy
from overrides import overrides
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_encoders import ComposeEncoder, FeedForwardEncoder, Seq2SeqEncoder
from allennlp.modules import FeedForward


class MockSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self, input_dim: int, output_dim: int, bidirectional: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bidirectional = bidirectional

    @overrides
    def forward(self, inputs, mask):
        pass

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional


def _make_feedforward(input_dim, output_dim):
    return FeedForwardEncoder(
        FeedForward(
            input_dim=input_dim, num_layers=1, activations=torch.nn.ReLU(), hidden_dims=output_dim
        )
    )


class TestPassThroughEncoder(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.encoder = ComposeEncoder(
            [_make_feedforward(9, 5), _make_feedforward(5, 10), _make_feedforward(10, 3)]
        )

    def test_get_dimension_is_correct(self):
        assert self.encoder.get_input_dim() == 9
        assert self.encoder.get_output_dim() == 3

    def test_composes(self):
        tensor = torch.zeros(2, 10, 9)
        output = self.encoder(tensor)

        for encoder in self.encoder.encoders:
            tensor = encoder(tensor)

        numpy.testing.assert_array_almost_equal(
            output.detach().cpu().numpy(), tensor.detach().cpu().numpy()
        )

    def test_pass_through_encoder_with_mask(self):
        tensor = torch.randn([2, 3, 9])
        mask = torch.tensor([[True, True, True], [True, False, False]])
        output = self.encoder(tensor, mask)

        for encoder in self.encoder.encoders:
            tensor = encoder(tensor, mask)

        numpy.testing.assert_array_almost_equal(
            output.detach().cpu().numpy(), tensor.detach().cpu().numpy()
        )

    def test_empty(self):
        with pytest.raises(ValueError):
            ComposeEncoder([])

    def test_mismatched_size(self):
        with pytest.raises(ValueError):
            ComposeEncoder(
                [
                    MockSeq2SeqEncoder(input_dim=9, output_dim=5),
                    MockSeq2SeqEncoder(input_dim=1, output_dim=2),
                ]
            )

    def test_mismatched_bidirectionality(self):
        with pytest.raises(ValueError):
            ComposeEncoder(
                [
                    MockSeq2SeqEncoder(input_dim=9, output_dim=5),
                    MockSeq2SeqEncoder(input_dim=5, output_dim=2, bidirectional=True),
                ]
            )

    def test_all_bidirectional(self):
        ComposeEncoder(
            [
                MockSeq2SeqEncoder(input_dim=9, output_dim=5, bidirectional=True),
                MockSeq2SeqEncoder(input_dim=5, output_dim=2, bidirectional=True),
            ]
        )
