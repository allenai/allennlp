# pylint: disable=no-self-use,invalid-name
import numpy
import pytest
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules import MultiHeadedMatrixAttention
from allennlp.modules.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules.seq2seq_encoders import IntraSentenceAttentionEncoder
from allennlp.modules.similarity_functions import MultiHeadedSimilarity


class TestIntraSentenceAttentionEncoder(AllenNlpTestCase):
    def test_get_dimension_is_correct(self):
        encoder = IntraSentenceAttentionEncoder(input_dim=5, projection_dim=4, combination='1,2')
        assert encoder.get_input_dim() == 5
        assert encoder.get_output_dim() == 9
        encoder = IntraSentenceAttentionEncoder(input_dim=5, combination='1+2')
        assert encoder.get_input_dim() == 5
        assert encoder.get_output_dim() == 5
        params = Params({'input_dim': 8,
                         'projection_dim': 8,
                         'num_attention_heads': 4,
                         'similarity_function': {'type': 'multiheaded_matrix_attention',
                                                 'num_heads': 4,
                                                 'tensor_1_dim': 8},
                         'combination': '1,2,1*2'})
        encoder = IntraSentenceAttentionEncoder.from_params(params)
        assert encoder.get_input_dim() == 8
        assert encoder.get_output_dim() == 24

    def test_constructor_asserts_multi_head_consistency(self):
        with pytest.raises(ConfigurationError) as exception_info:
            IntraSentenceAttentionEncoder(input_dim=5, num_attention_heads=4)
        assert 'Encoder has multiple heads' in exception_info.value.message
        similarity = LegacyMatrixAttention(MultiHeadedSimilarity(3, 6))
        with pytest.raises(ConfigurationError) as exception_info:
            IntraSentenceAttentionEncoder(input_dim=5, matrix_attention=similarity)
        assert 'Similarity function has multiple heads' in exception_info.value.message
        with pytest.raises(ConfigurationError) as exception_info:
            IntraSentenceAttentionEncoder(input_dim=5, num_attention_heads=2,
                                          matrix_attention=similarity)
        assert "Number of heads don't match" in exception_info.value.message

    def test_forward_works_with_simple_attention(self):
        # We're not going to check the output values here, as that's complicated; we'll just make
        # sure the code runs and the shapes are correct.
        encoder = IntraSentenceAttentionEncoder(input_dim=2)
        input_tensor = Variable(torch.from_numpy(numpy.random.rand(4, 3, 2)))
        encoder_output = encoder(input_tensor, None)
        assert list(encoder_output.size()) == [4, 3, 4]  # default combination is 1,2

    def test_forward_works_with_multi_headed_attention(self):
        # We're not going to check the output values here, as that's complicated; we'll just make
        # sure the code runs and the shapes are correct.
        similarity =  LegacyMatrixAttention(MultiHeadedSimilarity(3, 24))
        encoder = IntraSentenceAttentionEncoder(input_dim=24,
                                                projection_dim=24,
                                                matrix_attention=similarity,
                                                num_attention_heads=3,
                                                combination="1+2")
        input_tensor = Variable(torch.from_numpy(numpy.random.rand(4, 6, 24))).float()
        encoder_output = encoder(input_tensor, None)
        assert list(encoder_output.size()) == [4, 6, 24]
