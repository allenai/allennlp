import copy
import torch
import numpy as np
from allennlp.common import Params
from allennlp.modules.transformer import SinusoidalPositionalEncoding
from allennlp.common.testing import AllenNlpTestCase


class TestSinusoidalPositionalEncoding(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "min_timescale": 1.0,
            "max_timescale": 1.0e4,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.positional_encoding = SinusoidalPositionalEncoding.from_params(params)

    def test_can_construct_from_params(self):
        assert self.positional_encoding.min_timescale == self.params_dict["min_timescale"]
        assert self.positional_encoding.max_timescale == self.params_dict["max_timescale"]

    def test_forward(self):
        tensor2tensor_result = np.asarray(
            [
                [0.00000000e00, 0.00000000e00, 1.00000000e00, 1.00000000e00],
                [8.41470957e-01, 9.99999902e-05, 5.40302277e-01, 1.00000000e00],
                [9.09297407e-01, 1.99999980e-04, -4.16146845e-01, 1.00000000e00],
            ]
        )

        tensor = torch.zeros([2, 3, 4])
        result = self.positional_encoding(tensor)
        np.testing.assert_almost_equal(result[0].detach().cpu().numpy(), tensor2tensor_result)
        np.testing.assert_almost_equal(result[1].detach().cpu().numpy(), tensor2tensor_result)

        # Check case with odd number of dimensions.
        tensor2tensor_result = np.asarray(
            [
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    1.00000000e00,
                    1.00000000e00,
                    1.00000000e00,
                    0.00000000e00,
                ],
                [
                    8.41470957e-01,
                    9.99983307e-03,
                    9.99999902e-05,
                    5.40302277e-01,
                    9.99949992e-01,
                    1.00000000e00,
                    0.00000000e00,
                ],
                [
                    9.09297407e-01,
                    1.99986659e-02,
                    1.99999980e-04,
                    -4.16146815e-01,
                    9.99800026e-01,
                    1.00000000e00,
                    0.00000000e00,
                ],
            ]
        )

        tensor = torch.zeros([2, 3, 7])
        result = self.positional_encoding(tensor)
        np.testing.assert_almost_equal(result[0].detach().cpu().numpy(), tensor2tensor_result)
        np.testing.assert_almost_equal(result[1].detach().cpu().numpy(), tensor2tensor_result)
