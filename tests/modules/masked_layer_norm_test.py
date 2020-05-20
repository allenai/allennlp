import numpy as np
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.nn import util


class TestMaskedLayerNorm(AllenNlpTestCase):
    def test_masked_layer_norm(self):
        x_n = np.random.rand(2, 3, 7)
        mask_n = np.array([[1, 1, 0], [1, 1, 1]])

        x = torch.from_numpy(x_n).float()
        mask = torch.from_numpy(mask_n).bool()

        layer_norm = MaskedLayerNorm(7, gamma0=0.2)
        normed_x = layer_norm(x, mask)

        N = 7 * 5
        mean = (x_n * np.expand_dims(mask_n, axis=-1)).sum() / N
        std = np.sqrt(
            (((x_n - mean) * np.expand_dims(mask_n, axis=-1)) ** 2).sum() / N
            + util.tiny_value_of_dtype(torch.float)
        )
        expected = 0.2 * (x_n - mean) / (std + util.tiny_value_of_dtype(torch.float))

        assert np.allclose(normed_x.data.numpy(), expected)
