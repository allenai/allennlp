import torch
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.self_attentive_sentence_encoder import SelfAttentiveSentenceEncoder


class TestSelfAttentiveSentenceEncoder(AllenNlpTestCase):

    def test_forward_and_dimensions(self):
        self_attentive_encoder = SelfAttentiveSentenceEncoder(3, 4, 5, False)
        input_tensor = torch.randn(3, 5, 5)
        encoder_output = self_attentive_encoder(input_tensor)

        sentence_representation = encoder_output["representation"]
        attention_weights = encoder_output["attention"]

        # Test sentence representation shape
        assert list(sentence_representation.size()) == [3, 4 * 5]
        # Test attention weights shape
        assert list(attention_weights.size()) == [3, 5, 4]

    def test_forbenius_norm_presence(self):
        self_attentive_encoder = SelfAttentiveSentenceEncoder(3, 4, 5, True)
        input_tensor = torch.randn(3, 5, 5)
        encoder_output = self_attentive_encoder(input_tensor)
        assert "regularization_loss" in encoder_output

    def test_l2_matrix_norm_calculation(self):
        self_attentive_encoder = SelfAttentiveSentenceEncoder(3, 4, 5, False)

        input_matrix = torch.ones(3, 4, 4)

        norm = self_attentive_encoder.l2_matrix_norm(input_matrix)
        # Calculated manually
        assert norm == 12

    def test_masking_of_input(self):
        self_attentive_encoder = SelfAttentiveSentenceEncoder(3, 4, 5, False)
        input_tensor = torch.randn(3, 5, 5)
        mask = torch.ones(3, 5)
        mask[:, 3:] = 0

        encoder_output_with_mask = self_attentive_encoder.forward(input_tensor, mask)

        encoder_output_without_mask = self_attentive_encoder.forward(input_tensor[:, :3, :])
        attention = encoder_output_with_mask["attention"][0, :3, :].detach().cpu().numpy()
        attention_unmasked = encoder_output_without_mask["attention"][0, :, :].detach().cpu().numpy()

        numpy.testing.assert_almost_equal(attention, attention_unmasked)
        numpy.testing.assert_almost_equal(encoder_output_with_mask["representation"],
                                          encoder_output_without_mask["representation"])
