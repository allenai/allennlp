from typing import Dict, List, Tuple, Any
from collections import OrderedDict

import torch

from allennlp.training.optimizers import Optimizer
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.common.params import Params


def is_hat_shaped(learning_rates: List[float]):
    """
    Check if the list of learning rates is "hat" shaped, i.e.,
    increases then decreases
    """
    # sufficient conditions:
    #   has both an increasing and decreasing segment
    #   decrease segment occurs after increasing segment
    #   once start decreasing, can't increase again
    has_increasing_segment = False
    has_decreasing_segment = False
    for k in range(1, len(learning_rates)):
        delta = learning_rates[k] - learning_rates[k - 1]
        if delta > 1e-8:
            has_increasing_segment = True
            if has_decreasing_segment:
                # can't increase again after hitting the max
                return False
        elif delta < -1e-8:
            if not has_increasing_segment:
                # can't decrease without have an increasing segment
                return False
            has_decreasing_segment = True
        else:
            # no change
            pass

    return has_increasing_segment and has_decreasing_segment


class SlantedTriangularTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.model = torch.nn.Sequential(
            OrderedDict([("lin1", torch.nn.Linear(10, 10)), ("lin2", torch.nn.Linear(10, 10))])
        )

    def _get_optimizer(self, lr: float = 1.0):
        optimizer_params = Params({"type": "sgd", "lr": lr})
        optimizer_params["parameter_groups"] = [[[f"^{m}"], {}] for m in self.model._modules]
        return Optimizer.from_params(self.model.named_parameters(), optimizer_params)

    def _run_scheduler_get_lrs(self, params, num_steps_per_epoch):
        optimizer = self._get_optimizer()
        params["type"] = "slanted_triangular"
        scheduler = LearningRateScheduler.from_params(optimizer, Params(params))
        lrs = []

        batch_num_total = 0
        for epoch in range(params["num_epochs"]):
            for _ in range(num_steps_per_epoch):
                batch_num_total += 1
                # allennlp trainer calls step_batch after updating parameters
                # so collect lr at time of parameter update
                lrs.append(
                    [
                        param_group["lr"] * float(param_group["params"][0].requires_grad)
                        for param_group in optimizer.param_groups[:2]
                    ]
                )
                scheduler.step_batch(batch_num_total)
                if params.get("gradual_unfreezing") and epoch == 0:
                    assert scheduler.freezing_current
            # step() takes two arguments: validation metric and epoch
            scheduler.step(None, epoch)

        return lrs

    def test_is_hat_shaped(self):
        assert not is_hat_shaped([0.0] * 10)
        assert not is_hat_shaped([float(k) for k in range(10)])
        assert not is_hat_shaped([float(10 - k) for k in range(10)])
        assert is_hat_shaped([float(k) for k in range(10)] + [float(10 - k) for k in range(10)])
        assert not is_hat_shaped(
            [float(k) for k in range(10)]
            + [float(10 - k) for k in range(10)]
            + [float(k) for k in range(10)]
        )

    def test_from_params(self):
        optim = self._get_optimizer()
        sched = LearningRateScheduler.from_params(
            optim,
            Params(
                {
                    "type": "slanted_triangular",
                    "num_epochs": 5,
                    "num_steps_per_epoch": 10,
                    "gradual_unfreezing": True,
                    "discriminative_fine_tuning": True,
                    "decay_factor": 0.5,
                }
            ),
        )

        assert sched.num_epochs == 5
        assert sched.num_steps_per_epoch == 10
        assert sched.gradual_unfreezing is True
        assert sched.freezing_current is True

        assert len(optim.param_groups) == 3
        # The default parameter group in the Optimizer is empty
        assert not optim.param_groups[-1]["params"]
        assert optim.param_groups[-2]["lr"] == 1.0 / sched.ratio
        assert optim.param_groups[-3]["lr"] == 0.5 / sched.ratio

        with self.assertRaises(TypeError):
            # num_epochs and num_steps_per_epoch are required
            LearningRateScheduler.from_params(
                optim, Params({"type": "slanted_triangular", "num_epochs": 5})
            )
            LearningRateScheduler.from_params(
                optim, Params({"type": "slanted_triangular", "num_steps_epochs": 10})
            )

    def test_schedules(self):
        slanted_triangular_cases: List[Tuple[Dict[str, Any], List[Tuple[int, int, float]]]] = [
            (
                {
                    "num_epochs": 5,
                    "num_steps_per_epoch": 10,
                    "gradual_unfreezing": True,
                },  # parameters
                [
                    (0, 1, 0.03125),  # iteration, layer, learning rate
                    (0, 0, 0.0),
                    (1, 1, 1.0),
                    (1, 0, 0.0),
                    (9, 1, 0.138888),
                    (9, 0, 0.0),  # end of the first epoch
                    (10, 1, 0.03125),
                    (10, 0, 0.03125),
                    (14, 1, 1.0),
                    (14, 0, 1.0),
                    (49, 1, 0.05815972),
                    (49, 0, 0.05815972),
                ],
            ),
            (
                {
                    "num_epochs": 5,
                    "num_steps_per_epoch": 10,
                    "discriminative_fine_tuning": True,
                    "decay_factor": 0.5,
                },  # parameters
                [
                    (0, 1, 0.03125),  # iteration, layer, learning rate
                    (0, 0, 0.015625),
                    (5, 1, 1.0),
                    (5, 0, 0.5),
                    (49, 1, 0.052777),
                    (49, 0, 0.026388),
                ],
            ),
            (
                {
                    "num_epochs": 5,
                    "num_steps_per_epoch": 10,
                    "gradual_unfreezing": True,
                    "discriminative_fine_tuning": True,
                    "decay_factor": 0.5,
                },  # parameters
                [
                    (0, 1, 0.03125),  # iteration, layer, learning rate
                    (0, 0, 0.0),
                    (1, 1, 1.0),
                    (1, 0, 0.0),
                    (9, 1, 0.138888),
                    (9, 0, 0.0),  # end of the first epoch
                    (10, 1, 0.03125),
                    (10, 0, 0.015625),
                    (14, 1, 1.0),
                    (14, 0, 0.5),
                    (49, 1, 0.0581597222),
                    (49, 0, 0.0290798611),
                ],
            ),
        ]
        for params, lr_checks in slanted_triangular_cases:
            lrs = self._run_scheduler_get_lrs(params, params["num_steps_per_epoch"])

            for it, layer, lr in lr_checks:
                lr_check = round(lr, 5)
                lr = round(lrs[it][layer], 5)
                assert (
                    lr == lr_check
                ), f"Learning rate {lr} at iteration {it} at layer {layer} != {lr_check}."

    def test_schedules_num_steps_per_epoch(self):
        # ensure the learning rate schedule still maintains hat shape
        # if number of actual batches differs from parameter provided
        # in constructor
        for gradual_unfreezing in [True, False]:
            for discriminative_fine_tuning in [True, False]:
                for num_actual_steps_per_epoch in [7, 11]:
                    params = {
                        "num_epochs": 5,
                        "num_steps_per_epoch": 10,
                        "gradual_unfreezing": gradual_unfreezing,
                        "discriminative_fine_tuning": discriminative_fine_tuning,
                    }
                    lrs = self._run_scheduler_get_lrs(params, num_actual_steps_per_epoch)
                    first_layer_lrs = [rates[0] for rates in lrs]
                    second_layer_lrs = [rates[1] for rates in lrs]

                    if gradual_unfreezing:
                        assert max(first_layer_lrs[:num_actual_steps_per_epoch]) < 1e-8
                        assert min(first_layer_lrs[:num_actual_steps_per_epoch]) > -1e-8
                        assert is_hat_shaped(first_layer_lrs[num_actual_steps_per_epoch:])
                        assert is_hat_shaped(second_layer_lrs[:num_actual_steps_per_epoch])
                        assert is_hat_shaped(second_layer_lrs[num_actual_steps_per_epoch:])
                    else:
                        assert is_hat_shaped(first_layer_lrs)
                        assert is_hat_shaped(second_layer_lrs)
