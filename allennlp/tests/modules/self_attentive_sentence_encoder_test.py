# pylint: disable=no-self-use,invalid-name
import numpy
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.self_attentive_sentence_encoder import SelfAttentiveSentenceEncoder


class TestSelfAttentiveSentenceEncoder(AllenNlpTestCase):

    def test_forward_and_dimensions(self):
        self_attentive_encoder = SelfAttentiveSentenceEncoder(attention_size=3, num_attention_heads=4,
                                                              input_dim=5)
        input_tensor = torch.randn(3, 5, 5)
        encoder_output = self_attentive_encoder(input_tensor)

        sentence_representation = encoder_output["representation"]
        attention_weights = encoder_output["attention"]

        # Test sentence representation shape
        assert list(sentence_representation.size()) == [3, 4 * 5]
        # Test attention weights shape
        assert list(attention_weights.size()) == [3, 5, 4]

    def test_frobenius_norm_presence(self):
        self_attentive_encoder = SelfAttentiveSentenceEncoder(attention_size=3, num_attention_heads=4,
                                                              input_dim=5, regularization_coefficient=0.5)
        input_tensor = torch.randn(3, 5, 5)
        encoder_output = self_attentive_encoder(input_tensor)
        assert "regularization_loss" in encoder_output

    def test_regularization_penalty_calculation(self):
        self_attentive_encoder = SelfAttentiveSentenceEncoder(attention_size=3, num_attention_heads=4,
                                                              input_dim=5)

        input_matrix_identity = torch.eye(4, 5)
        input_matrix_identity = input_matrix_identity.unsqueeze(0).expand(3, 4, 5)
        input_matrix_zeros = torch.zeros(3, 4, 5)
        
        identity_norm = self_attentive_encoder.frobenius_regularization_penalty(input_matrix_identity)
        zeros_norm = self_attentive_encoder.frobenius_regularization_penalty(input_matrix_zeros)
        # Calculated manually
        assert identity_norm == 0
        assert zeros_norm == 6

    def test_masking_of_input(self):
        self_attentive_encoder = SelfAttentiveSentenceEncoder(attention_size=3, num_attention_heads=4,
                                                              input_dim=5)
        input_tensor = torch.randn(3, 5, 5)
        mask = torch.ones(3, 5)
        mask[:, 3:] = 0

        encoder_output_with_mask = self_attentive_encoder.forward(input_tensor, mask)

        encoder_output_without_mask = self_attentive_encoder.forward(input_tensor[:, :3, :])
        attention = encoder_output_with_mask["attention"][0, :3, :].detach().cpu().numpy()
        attention_unmasked = encoder_output_without_mask["attention"][0, :, :].detach().cpu().numpy()

        numpy.testing.assert_almost_equal(attention, attention_unmasked)
        numpy.testing.assert_almost_equal(encoder_output_with_mask["representation"].detach().cpu().numpy(),
                                          encoder_output_without_mask["representation"].detach().cpu().numpy(),
                                          decimal=6)
