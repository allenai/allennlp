# pylint: disable=no-self-use,invalid-name
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2vec_encoders import SelfAttentiveEncoderWrapper


class SelfAttentiveEncoderWrapperTest(AllenNlpTestCase):

    def test_get_dimensions_is_correct(self):
        self_attentive_encoder = SelfAttentiveEncoderWrapper(attention_size=5, num_attention_heads=3, 
                                                             input_dim=6)

        assert self_attentive_encoder.get_input_dim() == 6
        assert self_attentive_encoder.get_output_dim() == 18

    def test_forward_through_wrapper(self):
        self_attentive_encoder = SelfAttentiveEncoderWrapper(attention_size=5, num_attention_heads=3, 
                                                             input_dim=6)
        inputs = torch.randn(3, 5, 6)

        representation = self_attentive_encoder(inputs)

        assert list(representation.size()) == [3, 3*6]
