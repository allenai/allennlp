# pylint: disable=no-self-use,invalid-name,protected-access

from typing import Dict, Any

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.common.params import Params


class LearningRateSchedulersTest(AllenNlpTestCase):

    def test_reduce_on_plateau_error_throw_when_no_metrics_exist(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        with self.assertRaises(ConfigurationError) as context:
            LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                    Params({"type": "adam"})),
                                              Params({"type": "reduce_on_plateau"})).step(None, None)

        self.assertTrue(
                'The reduce_on_plateau learning rate scheduler requires a validation metric'
                in str(context.exception))

    def test_reduce_on_plateau_works_when_metrics_exist(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                Params({"type": "adam"})),
                                          Params({"type": "reduce_on_plateau"})).step(10, None)

    def test_no_metric_wrapper_can_support_none_for_metrics(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        lrs = LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                      Params({"type": "adam"})),
                                                Params({"type": "step", "step_size": 1}))
        lrs.step(None, None)

    def test_noam_learning_rate_schedule_does_not_crash(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        lrs = LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                      Params({"type": "adam"})),
                                                Params({"type": "noam", "model_size": 10, "warmup_steps": 2000}))
        lrs.step(None)
        lrs.step_batch(None)


class CosineWithRestartsTest(AllenNlpTestCase):

    def setUp(self):
        super(CosineWithRestartsTest, self).setUp()
        self.model = torch.nn.Sequential(torch.nn.Linear(10, 10))

        # We use these cases to verify that the scheduler works as expected.
        # Each case consists of 5 parameters:
        # - epochs: the total # of epochs to run for.
        # - params: parameters passed to initialize the scheduler.
        # - learning rate checks: a list of tuples, each of which specifies an epoch
        #   number and the expected value of the learning rate at that epoch.
        # - checkpoints: a list of epoch numbers at which to save the scheduler
        #   state, and then restore from the saved state and resume.
        self.cosine_schedule_cases = [
                (
                        30,
                        {"t_initial": 30, "t_mul": 1.0},
                        [(0, 1.0),
                         (15, 0.5000000000000001),
                         (29, 0.0027390523158632996)],
                        [10, 14]
                ),
                (
                        10,
                        {"t_initial": 1, "t_mul": 2.0},
                        [(0, 1.0),
                         (1, 1.0),
                         (2, 0.5),
                         (3, 1.0)],
                        [1, 3]
                ),
                (
                        30,
                        {"t_initial": 1, "t_mul": 1.0},
                        [(0, 1.0),
                         (15, 1.0),
                         (29, 1.0)],
                        []
                ),
                (
                        60,
                        {"t_initial": 30, "t_mul": 1.0},
                        [(0, 1.0),
                         (15, 0.5000000000000001),
                         (29, 0.0027390523158632996),
                         (30, 1.0),
                         (45, 0.5000000000000001),
                         (59, 0.0027390523158632996)],
                        [30, 35]
                ),
                (
                        60,
                        {"t_initial": 30, "t_mul": 1.0, "eta_mul": 0.5},
                        [(0, 1.0),
                         (15, 0.5000000000000001),
                         (29, 0.0027390523158632996),
                         (30, .5)],
                        []
                ),
                (
                        100,
                        {"t_initial": 30, "t_mul": 1.5},
                        [(0, 1.0),
                         (29, 0.0027390523158632996),
                         (30, 1.0),
                         (74, 0.0012179748700879012)],
                        []
                ),
                (
                        210,
                        {"t_initial": 30, "t_mul": 2},
                        [(0, 1.0),
                         (29, 0.0027390523158632996),
                         (30, 1.0),
                         (89, 0.0006852326227130834),
                         (90, 1.0),
                         (209, 0.00017133751222137006)],
                        []
                ),
                (
                        210,
                        {"t_initial": 30, "t_mul": 2, "eta_mul": 0.5},
                        [(0, 1.0),
                         (30, .5),
                         (90, .25)],
                        [29, 90]
                ),
                (
                        150,
                        {"t_initial": 30, "t_mul": 1},
                        [(0, 1.0),
                         (29, 0.0027390523158632996),
                         (30, 1.0),
                         (59, 0.0027390523158632996),
                         (60, 1.0),
                         (89, 0.0027390523158632996),
                         (90, 1.0)],
                        []
                ),
                (
                        10,
                        {"t_initial": 1, "t_mul": 1, "eta_mul": 0.5},
                        [(0, 1.0), (1, 0.5), (2, 0.25)],
                        []
                ),
        ]

    def _get_optimizer(self, lr: float = 1.0):
        return Optimizer.from_params(self.model.named_parameters(), Params({"type": "sgd", "lr": lr}))

    def test_from_params(self):
        """Make sure ``from_params`` initializes an instance properly."""
        optim = self._get_optimizer()
        sched = LearningRateScheduler.from_params(optim, Params({"type": "cosine", "t_initial": 5})).lr_scheduler

        assert sched.t_initial == 5
        assert sched._initialized is True

        # Learning should be unchanged after initializing scheduler.
        assert optim.param_groups[0]["lr"] == 1.0

        with self.assertRaises(TypeError):
            # t_initial is required.
            LearningRateScheduler.from_params(optim, Params({"type": "cosine"}))

    def test_schedules(self):
        """Make sure the math is correct."""
        for epochs, params, lr_checks, _ in self.cosine_schedule_cases:
            optimizer = self._get_optimizer()
            params["type"] = "cosine"
            scheduler = LearningRateScheduler.from_params(optimizer, Params(params)).lr_scheduler
            lrs = [optimizer.param_groups[0]["lr"]]
            for epoch in range(epochs):
                scheduler.step(epoch)
                lrs.append(optimizer.param_groups[0]["lr"])

            for it, lr in lr_checks:
                assert lrs[it] == lr

    def test_schedules_with_save_and_resume(self):
        """Make sure scheduler will resume with the right state."""

        def init_and_restore_scheduler(optimizer: torch.optim.Optimizer,
                                       params: Dict[str, Any],
                                       state_dict: Dict[str, Any] = None):
            """
            Initialize a new scheduler and optionally restore its state from
            a checkpoint.
            """
            params["type"] = "cosine"
            scheduler = LearningRateScheduler.from_params(optimizer, Params(params))
            if state_dict is not None:
                scheduler.lr_scheduler.load_state_dict(state_dict)
            return scheduler

        for epochs, params, lr_checks, checkpoints in self.cosine_schedule_cases:
            optimizer = self._get_optimizer()
            scheduler = init_and_restore_scheduler(optimizer, params)
            state = scheduler.lr_scheduler.state_dict()

            lrs = [optimizer.param_groups[0]["lr"]]
            for epoch in range(epochs):
                if epoch in checkpoints:
                    # Restore scheduler from state dict.
                    scheduler = init_and_restore_scheduler(optimizer, params, state_dict=state)

                # Take step and record learning rate.
                scheduler.step(1, epoch)
                lrs.append(optimizer.param_groups[0]["lr"])

                # Save state again.
                state = scheduler.lr_scheduler.state_dict()

            for it, lr in lr_checks:
                assert lrs[it] == lr
