import numpy
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.common.testing import AllenNlpTestCase


class TestBagOfEmbeddingsEncoder(AllenNlpTestCase):
    def test_get_dimension_is_correct(self):
        encoder = BagOfEmbeddingsEncoder(embedding_dim=5)
        assert encoder.get_input_dim() == 5
        assert encoder.get_output_dim() == 5
        encoder = BagOfEmbeddingsEncoder(embedding_dim=12)
        assert encoder.get_input_dim() == 12
        assert encoder.get_output_dim() == 12

    def test_can_construct_from_params(self):
        params = Params({"embedding_dim": 5})
        encoder = BagOfEmbeddingsEncoder.from_params(params)
        assert encoder.get_input_dim() == 5
        assert encoder.get_output_dim() == 5
        params = Params({"embedding_dim": 12, "averaged": True})
        encoder = BagOfEmbeddingsEncoder.from_params(params)
        assert encoder.get_input_dim() == 12
        assert encoder.get_output_dim() == 12

    def test_forward_does_correct_computation(self):
        encoder = BagOfEmbeddingsEncoder(embedding_dim=2)
        input_tensor = torch.FloatTensor(
            [[[0.7, 0.8], [0.1, 1.5], [0.3, 0.6]], [[0.5, 0.3], [1.4, 1.1], [0.3, 0.9]]]
        )
        mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]])
        encoder_output = encoder(input_tensor, mask)
        assert_almost_equal(
            encoder_output.data.numpy(),
            numpy.asarray([[0.7 + 0.1 + 0.3, 0.8 + 1.5 + 0.6], [0.5 + 1.4, 0.3 + 1.1]]),
        )

    def test_forward_does_correct_computation_with_average(self):
        encoder = BagOfEmbeddingsEncoder(embedding_dim=2, averaged=True)
        input_tensor = torch.FloatTensor(
            [
                [[0.7, 0.8], [0.1, 1.5], [0.3, 0.6]],
                [[0.5, 0.3], [1.4, 1.1], [0.3, 0.9]],
                [[0.4, 0.3], [0.4, 0.3], [1.4, 1.7]],
            ]
        )
        mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]])
        encoder_output = encoder(input_tensor, mask)
        assert_almost_equal(
            encoder_output.data.numpy(),
            numpy.asarray(
                [
                    [(0.7 + 0.1 + 0.3) / 3, (0.8 + 1.5 + 0.6) / 3],
                    [(0.5 + 1.4) / 2, (0.3 + 1.1) / 2],
                    [0.0, 0.0],
                ]
            ),
        )

    def test_forward_does_correct_computation_with_average_no_mask(self):
        encoder = BagOfEmbeddingsEncoder(embedding_dim=2, averaged=True)
        input_tensor = torch.FloatTensor(
            [[[0.7, 0.8], [0.1, 1.5], [0.3, 0.6]], [[0.5, 0.3], [1.4, 1.1], [0.3, 0.9]]]
        )
        encoder_output = encoder(input_tensor)
        assert_almost_equal(
            encoder_output.data.numpy(),
            numpy.asarray(
                [
                    [(0.7 + 0.1 + 0.3) / 3, (0.8 + 1.5 + 0.6) / 3],
                    [(0.5 + 1.4 + 0.3) / 3, (0.3 + 1.1 + 0.9) / 3],
                ]
            ),
        )
