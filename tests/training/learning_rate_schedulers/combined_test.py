import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import (
    LearningRateScheduler,
    CombinedLearningRateScheduler,
    PolynomialDecay,
)
from allennlp.training.optimizers import Optimizer


class TestCombinedLRScheduler(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        self.optimizer = Optimizer.from_params(
            model_parameters=self.model.named_parameters(),
            params=Params({"type": "sgd", "lr": 1.0}),
        )

    def get_scheduler(self) -> LearningRateScheduler:
        return LearningRateScheduler.from_params(
            Params(
                {
                    "type": "combined",
                    "schedulers": [
                        [
                            2,
                            {
                                "type": "polynomial_decay",
                                "warmup_steps": 10,
                                "end_learning_rate": 0.5,
                            },
                        ],
                        [
                            5,
                            {
                                "type": "polynomial_decay",
                                "warmup_steps": 0,
                                "end_learning_rate": 0.1,
                            },
                        ],
                    ],
                }
            ),
            optimizer=self.optimizer,
            num_steps_per_epoch=10,
        )

    def test_partial_schedule(self):
        scheduler = self.get_scheduler()
        assert isinstance(scheduler, CombinedLearningRateScheduler)
        assert isinstance(scheduler._current_scheduler, PolynomialDecay)

        # This should be 0 because the PolynomialDecay scheduler initializes the LR to 0.
        assert self.optimizer.param_groups[0]["lr"] == 0.0

        epoch_end_lrs = []
        for epoch in range(10):
            if epoch > 6:
                assert scheduler._current_scheduler is None
            elif epoch >= 2:
                assert scheduler._current_scheduler is not None
                assert scheduler._current_scheduler.total_steps == 50
                assert scheduler._current_scheduler.base_values[0] == 0.5
            else:
                assert scheduler._current_scheduler is not None
                assert scheduler._current_scheduler.total_steps == 20
                assert scheduler._current_scheduler.base_values[0] == 1.0

            for step in range(10):
                scheduler.step_batch()

            scheduler.step()

            epoch_end_lrs.append(self.optimizer.param_groups[0]["lr"])

        assert epoch_end_lrs[0] == 1.0
        assert epoch_end_lrs[1] == 0.5
        assert epoch_end_lrs[6] == 0.1
        assert epoch_end_lrs[6] == 0.1

    def test_load_from_checkpoint(self):
        scheduler = self.get_scheduler()

        for epoch in range(3):
            for step in range(10):
                scheduler.step_batch()
            scheduler.step()

        assert scheduler.last_epoch == 2
        assert scheduler._current_scheduler is not None
        assert scheduler._current_scheduler.total_steps == 50
        assert scheduler._current_scheduler.base_values[0] == 0.5

        state_dict = scheduler.state_dict()
        new_scheduler = self.get_scheduler()
        new_scheduler.load_state_dict(state_dict)

        assert new_scheduler.last_epoch == 2
        assert new_scheduler._current_scheduler is not None
        assert new_scheduler._current_scheduler.total_steps == 50
        assert new_scheduler._current_scheduler.base_values[0] == 0.5, state_dict
