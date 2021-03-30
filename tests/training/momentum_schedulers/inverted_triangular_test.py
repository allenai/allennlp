from math import isclose
import torch

from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.optimizers import Optimizer


class InvertedTriangularTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        self.base_momentum = 0.9

    def _get_optimizer(self):
        return Optimizer.from_params(
            model_parameters=self.model.named_parameters(),
            params=Params({"type": "sgd", "lr": 1.0, "momentum": self.base_momentum}),
        )

    def test_from_params(self):
        optimizer = self._get_optimizer()
        scheduler = MomentumScheduler.from_params(
            optimizer=optimizer,
            params=Params({"type": "inverted_triangular", "cool_down": 10, "warm_up": 10}),
        )
        assert scheduler.cool_down == 10
        assert scheduler.warm_up == 10
        assert scheduler.ratio == 10
        assert scheduler.last_epoch == -1

    def test_basic_schedule(self):
        optimizer = self._get_optimizer()
        scheduler = MomentumScheduler.from_params(
            optimizer=optimizer,
            params=Params(
                {"type": "inverted_triangular", "cool_down": 6, "warm_up": 10, "ratio": 5}
            ),
        )
        # Before first epoch, momentum should be unchanged.
        assert optimizer.param_groups[0]["momentum"] == self.base_momentum
        # After first epoch, `step` is called, and momentum should be adjusted for
        # the next epoch.
        scheduler.step()
        assert isclose(
            optimizer.param_groups[0]["momentum"],
            self.base_momentum - (self.base_momentum - self.base_momentum / 5) * (1 / 6),
        )
        # After second epoch, `step` is called and momentum is updated for 3rd epoch.
        scheduler.step()
        assert isclose(
            optimizer.param_groups[0]["momentum"],
            self.base_momentum - (self.base_momentum - self.base_momentum / 5) * (2 / 6),
        )
        scheduler.last_epoch = 4
        # ... after the 6th epoch (epoch id 5), momentum should be set to `base_momentum / ratio`.
        scheduler.step()
        assert isclose(optimizer.param_groups[0]["momentum"], self.base_momentum / 5)
        # Then the momentum stars increasing again.
        scheduler.step()
        assert isclose(
            optimizer.param_groups[0]["momentum"],
            self.base_momentum / 5 + (self.base_momentum - self.base_momentum / 5) * (1 / 10),
        )
        # After the 16th epoch (6 + 10) (epoch id 15), momentum should be back to the base level.
        scheduler.last_epoch = 14
        scheduler.step()
        assert isclose(optimizer.param_groups[0]["momentum"], self.base_momentum)
        scheduler.step()
        assert isclose(optimizer.param_groups[0]["momentum"], self.base_momentum)
        scheduler.step()
        assert isclose(optimizer.param_groups[0]["momentum"], self.base_momentum)
