from typing import Dict

import torch
import numpy as np

from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.model import Model
from allennlp.training.moving_average import MovingAverage, ExponentialMovingAverage


class MovingAverageTest(AllenNlpTestCase):
    def test_from_params(self):
        params = Params({"type": "exponential", "decay": 0.99})

        _ = MovingAverage.from_params(params, parameters=[])

    def test_exponential_moving_average_without_steps(self):
        param1 = torch.ones(5, 3)
        param2 = torch.ones(2)
        moving_average = ExponentialMovingAverage(
            [("param1", param1), ("param2", param2)], decay=0.9999
        )

        param1.data *= 5  # now all 5s
        param2.data *= 10  # now all 10s
        moving_average.apply()

        param1.data *= 5  # now all 25s
        param2.data *= 10  # now all 100s
        moving_average.apply()

        # Get shadow variables
        moving_average.assign_average_value()

        np.testing.assert_array_almost_equal(
            param1, 1 * 0.9999 ** 2 + 5 * 0.9999 * 0.0001 + 25 * 0.0001
        )
        np.testing.assert_array_almost_equal(
            param2, 1 * 0.9999 ** 2 + 10 * 0.9999 * 0.0001 + 100 * 0.0001
        )

        # Restore original variables
        moving_average.restore()
        np.testing.assert_array_almost_equal(param1, 25)
        np.testing.assert_array_almost_equal(param2, 100)

    def test_exponential_moving_average_num_updates(self):
        param1 = torch.ones(5, 3)
        param2 = torch.ones(2)
        moving_average = ExponentialMovingAverage(
            [("param1", param1), ("param2", param2)], decay=0.9999
        )

        param1.data *= 5  # now all 5s
        param2.data *= 10  # now all 10s
        moving_average.apply(num_updates=100)  # 101 / 110 ~ 0.92 < 0.9999

        param1.data *= 5  # now all 25s
        param2.data *= 10  # now all 100s
        moving_average.apply(num_updates=1_000_000)  # 1_000_001 / 1_000_010 ~ .999991 > .9999

        # Get shadow variables
        moving_average.assign_average_value()

        np.testing.assert_array_almost_equal(
            param1, 1 * (101 / 110) * 0.9999 + 5 * (9 / 110) * 0.9999 + 25 * 0.0001
        )

        np.testing.assert_array_almost_equal(
            param2, 1 * (101 / 110) * 0.9999 + 10 * (9 / 110) * 0.9999 + 100 * 0.0001
        )

        # Restore original variables
        moving_average.restore()
        np.testing.assert_array_almost_equal(param1, 25)
        np.testing.assert_array_almost_equal(param2, 100)

    def test_works_with_model(self):
        class FakeModel(Model):
            def __init__(self) -> None:
                super().__init__(None)
                self.w = torch.nn.Parameter(torch.randn(1))

            def forward(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore
                return {"loss": (t * self.w).sum()}

        model = FakeModel()
        moving_average = ExponentialMovingAverage(model.named_parameters())

        optimizer = torch.optim.SGD(list(model.parameters()), lr=0.1)

        for _ in range(10):
            optimizer.zero_grad()
            t = torch.randn(10)
            loss = model.forward(t)["loss"]
            loss.backward()
            optimizer.step()
            moving_average.apply()

        w_value = model.w.item()
        shadow_value = moving_average._shadows["w"].item()

        assert w_value != shadow_value

        moving_average.assign_average_value()

        assert model.w.item() == shadow_value

        moving_average.restore()

        assert model.w.item() == w_value

        # Now keep training:

        for _ in range(10):
            optimizer.zero_grad()
            t = torch.randn(10)
            loss = model.forward(t)["loss"]
            loss.backward()
            optimizer.step()
            moving_average.apply()
