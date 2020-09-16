from typing import Any, Dict, List, Tuple, Union
import torch
from torch.testing import assert_allclose

from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    global_distributed_metric,
    run_distributed_test,
)
from allennlp.training.metrics import SequenceAccuracy


class SequenceAccuracyTest(AllenNlpTestCase):
    @multi_device
    def test_sequence_accuracy(self, device: str):
        accuracy = SequenceAccuracy()
        gold = torch.tensor([[1, 2, 3], [2, 4, 8], [0, 1, 1]], device=device)
        predictions = torch.tensor(
            [[[1, 2, 3], [1, 2, -1]], [[2, 4, 8], [2, 5, 9]], [[-1, -1, -1], [0, 1, -1]]],
            device=device,
        )

        accuracy(predictions, gold)
        actual_accuracy = accuracy.get_metric()["accuracy"]
        assert_allclose(actual_accuracy, 2 / 3)

    @multi_device
    def test_sequence_accuracy_respects_mask(self, device: str):
        accuracy = SequenceAccuracy()
        gold = torch.tensor([[1, 2, 3], [2, 4, 8], [0, 1, 1], [11, 13, 17]], device=device)
        predictions = torch.tensor(
            [
                [[1, 2, 3], [1, 2, -1]],
                [[2, 4, 8], [2, 5, 9]],
                [[-1, -1, -1], [0, 1, -1]],
                [[12, 13, 17], [11, 13, 18]],
            ],
            device=device,
        )
        mask = torch.tensor(
            [[False, True, True], [True, True, True], [True, True, False], [True, False, True]],
            device=device,
        )

        accuracy(predictions, gold, mask)
        actual_accuracy = accuracy.get_metric()["accuracy"]
        assert_allclose(actual_accuracy, 3 / 4)

    @multi_device
    def test_sequence_accuracy_accumulates_and_resets_correctly(self, device: str):
        accuracy = SequenceAccuracy()
        gold = torch.tensor([[1, 2, 3]], device=device)
        accuracy(torch.tensor([[[1, 2, 3]]], device=device), gold)
        accuracy(torch.tensor([[[1, 2, 4]]], device=device), gold)

        actual_accuracy = accuracy.get_metric(reset=True)["accuracy"]
        assert_allclose(actual_accuracy, 1 / 2)
        assert accuracy.correct_count == 0
        assert accuracy.total_count == 0

    @multi_device
    def test_get_metric_on_new_object_works(self, device: str):
        accuracy = SequenceAccuracy()

        actual_accuracy = accuracy.get_metric(reset=True)["accuracy"]
        assert_allclose(actual_accuracy, 0)

    def test_distributed_sequence_accuracy(self):
        gold = torch.tensor([[1, 2, 3], [2, 4, 8], [0, 1, 1], [11, 13, 17]])
        predictions = torch.tensor(
            [
                [[1, 2, 3], [1, 2, -1]],
                [[2, 4, 8], [2, 5, 9]],
                [[-1, -1, -1], [0, 1, -1]],
                [[12, 13, 17], [11, 13, 18]],
            ]
        )
        mask = torch.tensor(
            [[False, True, True], [True, True, True], [True, True, False], [True, False, True]],
        )
        gold = [gold[:2], gold[2:]]
        predictions = [predictions[:2], predictions[2:]]
        mask = [mask[:2], mask[2:]]

        metric_kwargs = {"predictions": predictions, "gold_labels": gold, "mask": mask}
        desired_values = {"accuracy": 3 / 4}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            SequenceAccuracy(),
            metric_kwargs,
            desired_values,
            exact=False,
        )

    def test_multiple_distributed_runs(self):
        gold = torch.tensor([[1, 2, 3], [2, 4, 8], [0, 1, 1], [11, 13, 17]])
        predictions = torch.tensor(
            [
                [[1, 2, 3], [1, 2, -1]],
                [[2, 4, 8], [2, 5, 9]],
                [[-1, -1, -1], [0, 1, -1]],
                [[12, 13, 17], [11, 13, 18]],
            ]
        )
        mask = torch.tensor(
            [[False, True, True], [True, True, True], [True, True, False], [True, False, True]],
        )
        gold = [gold[:2], gold[2:]]
        predictions = [predictions[:2], predictions[2:]]
        mask = [mask[:2], mask[2:]]

        metric_kwargs = {"predictions": predictions, "gold_labels": gold, "mask": mask}
        desired_values = {"accuracy": 3 / 4}
        run_distributed_test(
            [-1, -1],
            multiple_runs,
            SequenceAccuracy(),
            metric_kwargs,
            desired_values,
            exact=True,
        )


def multiple_runs(
    global_rank: int,
    world_size: int,
    gpu_id: Union[int, torch.device],
    metric: SequenceAccuracy,
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

    assert desired_values["accuracy"] == metric.get_metric()["accuracy"]
