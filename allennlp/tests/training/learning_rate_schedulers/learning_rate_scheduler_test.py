import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.common.params import Params


class LearningRateSchedulersTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.model = torch.nn.Sequential(torch.nn.Linear(10, 10))

    def test_reduce_on_plateau_error_throw_when_no_metrics_exist(self):
        with self.assertRaises(ConfigurationError) as context:
            LearningRateScheduler.from_params(
                optimizer=Optimizer.from_params(
                    model_parameters=self.model.named_parameters(), params=Params({"type": "adam"})
                ),
                params=Params({"type": "reduce_on_plateau"}),
            ).step(None)
        assert "learning rate scheduler requires a validation metric" in str(context.exception)

    def test_reduce_on_plateau_works_when_metrics_exist(self):
        LearningRateScheduler.from_params(
            optimizer=Optimizer.from_params(
                model_parameters=self.model.named_parameters(), params=Params({"type": "adam"})
            ),
            params=Params({"type": "reduce_on_plateau"}),
        ).step(10)

    def test_no_metric_wrapper_can_support_none_for_metrics(self):
        lrs = LearningRateScheduler.from_params(
            optimizer=Optimizer.from_params(
                model_parameters=self.model.named_parameters(), params=Params({"type": "adam"})
            ),
            params=Params({"type": "step", "step_size": 1}),
        )
        lrs.lr_scheduler.optimizer.step()  # to avoid a pytorch warning
        lrs.step(None)

    def test_noam_learning_rate_schedule_does_not_crash(self):
        lrs = LearningRateScheduler.from_params(
            optimizer=Optimizer.from_params(
                model_parameters=self.model.named_parameters(), params=Params({"type": "adam"})
            ),
            params=Params({"type": "noam", "model_size": 10, "warmup_steps": 2000}),
        )
        lrs.step(None)
        lrs.step_batch(None)

    def test_exponential_works_properly(self):
        scheduler = LearningRateScheduler.from_params(
            optimizer=Optimizer.from_params(
                model_parameters=self.model.named_parameters(),
                params=Params({"type": "sgd", "lr": 1.0}),
            ),
            params=Params({"type": "exponential", "gamma": 0.5}),
        )
        optimizer = scheduler.lr_scheduler.optimizer
        optimizer.step()  # to avoid a pytorch warning
        # Initial learning rate should be unchanged for first epoch.
        assert optimizer.param_groups[0]["lr"] == 1.0
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == 0.5
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == 0.5 ** 2
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == 0.5 ** 3
