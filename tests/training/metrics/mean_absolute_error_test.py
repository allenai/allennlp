import torch

from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    global_distributed_metric,
    run_distributed_test,
)
from allennlp.training.metrics import MeanAbsoluteError


class MeanAbsoluteErrorTest(AllenNlpTestCase):
    @multi_device
    def test_mean_absolute_error_computation(self, device: str):
        mae = MeanAbsoluteError()
        predictions = torch.tensor(
            [[1.0, 1.5, 1.0], [2.0, 3.0, 3.5], [4.0, 5.0, 5.5], [6.0, 7.0, 7.5]], device=device
        )
        targets = torch.tensor(
            [[0.0, 1.0, 0.0], [2.0, 2.0, 0.0], [4.0, 5.0, 0.0], [7.0, 7.0, 0.0]], device=device
        )
        mae(predictions, targets)
        assert mae.get_metric()["mae"] == 21.0 / 12.0

        mask = torch.tensor(
            [[True, True, False], [True, True, False], [True, True, False], [True, True, False]],
            device=device,
        )
        mae(predictions, targets, mask)
        assert mae.get_metric()["mae"] == (21.0 + 3.5) / (12.0 + 8.0)

        new_targets = torch.tensor(
            [[2.0, 2.0, 0.0], [0.0, 1.0, 0.0], [7.0, 7.0, 0.0], [4.0, 5.0, 0.0]], device=device
        )
        mae(predictions, new_targets)
        assert mae.get_metric()["mae"] == (21.0 + 3.5 + 32.0) / (12.0 + 8.0 + 12.0)

        mae.reset()
        mae(predictions, new_targets)
        assert mae.get_metric()["mae"] == 32.0 / 12.0

    def test_distributed_accuracy(self):
        predictions = [
            torch.tensor([[1.0, 1.5, 1.0], [2.0, 3.0, 3.5]]),
            torch.tensor([[4.0, 5.0, 5.5], [6.0, 7.0, 7.5]]),
        ]
        targets = [
            torch.tensor([[0.0, 1.0, 0.0], [2.0, 2.0, 0.0]]),
            torch.tensor([[4.0, 5.0, 0.0], [7.0, 7.0, 0.0]]),
        ]
        metric_kwargs = {"predictions": predictions, "gold_labels": targets}
        desired_values = {"mae": 21.0 / 12.0}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            MeanAbsoluteError(),
            metric_kwargs,
            desired_values,
            exact=True,
        )
