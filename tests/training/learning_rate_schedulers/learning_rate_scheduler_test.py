import torch
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.common.params import Params


class LearningRateSchedulersTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.model = torch.nn.Sequential(torch.nn.Linear(10, 10))

    def test_reduce_on_plateau_error_throw_when_no_metrics_exist(self):
        with pytest.raises(
            ConfigurationError, match="learning rate scheduler requires a validation metric"
        ):
            LearningRateScheduler.from_params(
                optimizer=Optimizer.from_params(
                    model_parameters=self.model.named_parameters(), params=Params({"type": "adam"})
                ),
                params=Params({"type": "reduce_on_plateau"}),
            ).step(None)

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

    def test_polynomial_decay_works_properly(self):
        scheduler = LearningRateScheduler.from_params(
            optimizer=Optimizer.from_params(
                model_parameters=self.model.named_parameters(),
                params=Params({"type": "sgd", "lr": 1.0}),
            ),
            params=Params(
                {
                    "type": "polynomial_decay",
                    "warmup_steps": 2,
                    "num_epochs": 2,
                    "num_steps_per_epoch": 3,
                    "end_learning_rate": 0.1,
                    "power": 2,
                }
            ),
        )
        optimizer = scheduler.optimizer

        # Linear warmup for 2 steps.
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 0.5  # 1.0 * 1/2
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 1.0  # 1.0 * 2/2

        # Polynomial decay for 4 steps.
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 0.60625  # (1.0 - 0.1) * (3/4) ** 2 + 0.1
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 0.325  # (1.0 - 0.1) * (2/4) ** 2 + 0.1
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 0.15625  # (1.0 - 0.1) * (1/4) ** 2 + 0.1
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 0.1  # (1.0 - 0.1) * (0/4) ** 2 + 0.1

    def test_linear_with_warmup_works_properly(self):
        scheduler = LearningRateScheduler.from_params(
            optimizer=Optimizer.from_params(
                model_parameters=self.model.named_parameters(),
                params=Params({"type": "sgd", "lr": 1.0}),
            ),
            params=Params(
                {
                    "type": "linear_with_warmup",
                    "warmup_steps": 2,
                    "num_epochs": 2,
                    "num_steps_per_epoch": 3,
                }
            ),
        )
        optimizer = scheduler.optimizer

        # Linear warmup for 2 steps.
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 0.5  # 1.0 * 1/2
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 1.0  # 1.0 * 2/2

        # Linear decay for 4 steps.
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 0.75
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 0.5
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 0.25
        scheduler.step_batch()
        assert optimizer.param_groups[0]["lr"] == 0.0

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
        assert optimizer.param_groups[0]["lr"] == 0.5**2
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == 0.5**3

    def test_huggingface_schedulers_work_properly(self):
        def unwrap_schedule(scheduler, num_steps=10):
            lrs = []
            for _ in range(num_steps):
                lrs.append(scheduler.lr_scheduler.optimizer.param_groups[0]["lr"])
                scheduler.step()
            return lrs

        common_kwargs = {"num_warmup_steps": 2, "num_training_steps": 10}
        # schedulers doct format
        # function: (sched_args_dict, expected_learning_rates)
        scheds = {
            "constant": ({}, [10.0] * 10),
            "constant_with_warmup": (
                {"num_warmup_steps": 4},
                [0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            ),
            "cosine_with_warmup": (
                {**common_kwargs},
                [0.0, 5.0, 10.0, 9.61, 8.53, 6.91, 5.0, 3.08, 1.46, 0.38],
            ),
            "cosine_hard_restarts_with_warmup": (
                {**common_kwargs, "num_cycles": 2},
                [0.0, 5.0, 10.0, 8.53, 5.0, 1.46, 10.0, 8.53, 5.0, 1.46],
            ),
        }

        for scheduler_func, data in scheds.items():
            kwargs, expected_learning_rates = data

            scheduler = LearningRateScheduler.from_params(
                optimizer=Optimizer.from_params(
                    model_parameters=self.model.named_parameters(),
                    params=Params({"type": "adam", "lr": 10.0}),
                ),
                params=Params({"type": scheduler_func, **kwargs}),
            )
            optimizer = scheduler.lr_scheduler.optimizer
            optimizer.step()  # to avoid a pytorch warning

            lrs = unwrap_schedule(scheduler, 10)
            assert lrs == pytest.approx(
                expected_learning_rates, abs=1e-2
            ), f"failed for {scheduler_func} in normal scheduler"
