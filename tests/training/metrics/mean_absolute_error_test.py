from typing import Any, Dict, List, Tuple, Union
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

    def test_multiple_distributed_runs(self):
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
            multiple_runs,
            MeanAbsoluteError(),
            metric_kwargs,
            desired_values,
            exact=True,
        )


def multiple_runs(
    global_rank: int,
    world_size: int,
    gpu_id: Union[int, torch.device],
    metric: MeanAbsoluteError,
    metric_kwargs: Dict[str, List[Any]],
    desired_values: Dict[str, Any],
    exact: Union[bool, Tuple[float, float]] = True,
):

    kwargs = {}
    # Use the arguments meant for the process with rank `global_rank`.
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]

    for i in range(200):
        metric(**kwargs)

    assert desired_values["mae"] == metric.get_metric()["mae"]
