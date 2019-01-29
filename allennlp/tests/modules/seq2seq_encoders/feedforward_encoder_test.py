# pylint: disable=no-self-use,invalid-name
import torch
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders.feedforward_encoder import FeedForwardEncoder
from allennlp.nn import Activation


class TestFeedforwardEncoder(AllenNlpTestCase):
    def test_get_dimension_is_correct(self):
        feedforward = FeedForward(input_dim=10,
                                  num_layers=1,
                                  hidden_dims=10,
                                  activations="linear")
        encoder = FeedForwardEncoder(feedforward)
        assert encoder.get_input_dim() == feedforward.get_input_dim()
        assert encoder.get_output_dim() == feedforward.get_output_dim()

    def test_feedforward_encoder_exactly_match_feedforward_each_item(self):
        feedforward = FeedForward(input_dim=10,
                                  num_layers=1,
                                  hidden_dims=10,
                                  activations=Activation.by_name("linear")())
        encoder = FeedForwardEncoder(feedforward)
        tensor = torch.randn([2, 3, 10])
        output = encoder(tensor)
        target = feedforward(tensor)
        numpy.testing.assert_array_almost_equal(target.detach().cpu().numpy(),
                                                output.detach().cpu().numpy())

        # mask should work
        mask = torch.LongTensor([[1, 1, 1], [1, 0, 0]])
        output = encoder(tensor, mask)
        target = feedforward(tensor)  * mask.unsqueeze(dim=-1).float()
        numpy.testing.assert_array_almost_equal(target.detach().cpu().numpy(),
                                                output.detach().cpu().numpy())
