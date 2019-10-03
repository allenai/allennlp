from numpy.testing import assert_almost_equal
import torch

from allennlp.modules import ResidualWithLayerDropout
from allennlp.common.testing import AllenNlpTestCase


class TestResidualWithLayerDropout(AllenNlpTestCase):
    def test_dropout_works_for_training(self):
        layer_input_tensor = torch.FloatTensor([[2, 1], [-3, -2]])
        layer_output_tensor = torch.FloatTensor([[1, 3], [2, -1]])

        # The layer output should be dropped
        residual_with_layer_dropout = ResidualWithLayerDropout(1)
        residual_with_layer_dropout.train()
        result = residual_with_layer_dropout(layer_input_tensor, layer_output_tensor).data.numpy()
        assert result.shape == (2, 2)
        assert_almost_equal(result, [[2, 1], [-3, -2]])

        result = residual_with_layer_dropout(
            layer_input_tensor, layer_output_tensor, 1, 1
        ).data.numpy()
        assert result.shape == (2, 2)
        assert_almost_equal(result, [[2, 1], [-3, -2]])

        # The layer output should not be dropped
        residual_with_layer_dropout = ResidualWithLayerDropout(0.0)
        residual_with_layer_dropout.train()
        result = residual_with_layer_dropout(layer_input_tensor, layer_output_tensor).data.numpy()
        assert result.shape == (2, 2)
        assert_almost_equal(result, [[2 + 1, 1 + 3], [-3 + 2, -2 - 1]])

    def test_dropout_works_for_testing(self):
        layer_input_tensor = torch.FloatTensor([[2, 1], [-3, -2]])
        layer_output_tensor = torch.FloatTensor([[1, 3], [2, -1]])

        # During testing, the layer output is re-calibrated according to the survival probability,
        # and then added to the input.
        residual_with_layer_dropout = ResidualWithLayerDropout(0.2)
        residual_with_layer_dropout.eval()
        result = residual_with_layer_dropout(layer_input_tensor, layer_output_tensor).data.numpy()
        assert result.shape == (2, 2)
        assert_almost_equal(result, [[2 + 1 * 0.8, 1 + 3 * 0.8], [-3 + 2 * 0.8, -2 - 1 * 0.8]])
