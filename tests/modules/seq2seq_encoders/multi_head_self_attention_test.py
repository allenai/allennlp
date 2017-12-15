# pylint: disable=invalid-name,no-self-use,too-many-public-methods
import numpy
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_encoders import MultiHeadSelfAttention


class MultiHeadSelfAttentionTest(AllenNlpTestCase):
    def test_multi_head_self_attention_runs_forward(self):
        attention = MultiHeadSelfAttention(num_heads=3,
                                           input_dim=5,
                                           attention_dim=7,
                                           values_dim=9)
        inputs = Variable(torch.randn(2, 12, 5))
        assert list(attention(inputs).size()) == [2, 12, 5]

    def test_multi_head_self_attention_respects_masking(self):
        attention = MultiHeadSelfAttention(num_heads=3,
                                           input_dim=5,
                                           attention_dim=7,
                                           values_dim=9,
                                           attention_dropout_prob=0.0)
        tensor = Variable(torch.randn(2, 12, 5))
        mask = Variable(torch.ones([2, 12]))
        mask[0, 6:] = 0
        result = attention(tensor, mask)
        # Compute the same function without a mask, but with
        # only the unmasked elements - should be the same.
        result_without_mask = attention(tensor[:, :6, :])
        numpy.testing.assert_almost_equal(result[0, :6, :].data.cpu().numpy(),
                                          result_without_mask[0, :, :].data.cpu().numpy())
