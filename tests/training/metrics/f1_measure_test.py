import pytest
import torch
from torch.testing import assert_allclose

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    run_distributed_test,
    global_distributed_metric,
)
from allennlp.training.metrics import F1Measure


class F1MeasureTest(AllenNlpTestCase):
    @multi_device
    def test_f1_measure_catches_exceptions(self, device: str):
        f1_measure = F1Measure(0)
        predictions = torch.rand([5, 7], device=device)
        out_of_range_labels = torch.tensor([10, 3, 4, 0, 1], device=device)
        with pytest.raises(ConfigurationError):
            f1_measure(predictions, out_of_range_labels)

    @multi_device
    def test_f1_measure(self, device: str):
        f1_measure = F1Measure(positive_label=0)
        predictions = torch.tensor(
            [
                [0.35, 0.25, 0.1, 0.1, 0.2],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.5, 0.1, 0.2, 0.0],
                [0.1, 0.2, 0.1, 0.7, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
            ],
            device=device,
        )
        # [True Positive, True Negative, True Negative,
        #  False Negative, True Negative, False Negative]
        targets = torch.tensor([0, 4, 1, 0, 3, 0], device=device)
        f1_measure(predictions, targets)
        metrics = f1_measure.get_metric()
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        assert f1_measure._true_positives == 1.0
        assert f1_measure._true_negatives == 3.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 2.0
        f1_measure.reset()
        # check value
        assert_allclose(precision, 1.0)
        assert_allclose(recall, 0.333333333)
        assert_allclose(f1, 0.499999999)
        # check type
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(f1, float)

        # Test the same thing with a mask:
        mask = torch.tensor([True, False, True, True, True, False], device=device)
        f1_measure(predictions, targets, mask)
        metrics = f1_measure.get_metric()
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        assert f1_measure._true_positives == 1.0
        assert f1_measure._true_negatives == 2.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 1.0
        f1_measure.reset()
        assert_allclose(precision, 1.0)
        assert_allclose(recall, 0.5)
        assert_allclose(f1, 0.6666666666)

    @multi_device
    def test_f1_measure_other_positive_label(self, device: str):
        f1_measure = F1Measure(positive_label=1)
        predictions = torch.tensor(
            [
                [0.35, 0.25, 0.1, 0.1, 0.2],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.5, 0.1, 0.2, 0.0],
                [0.1, 0.2, 0.1, 0.7, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
            ],
            device=device,
        )
        # [True Negative, False Positive, True Positive,
        #  False Positive, True Negative, False Positive]
        targets = torch.tensor([0, 4, 1, 0, 3, 0], device=device)
        f1_measure(predictions, targets)
        metrics = f1_measure.get_metric()
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        assert f1_measure._true_positives == 1.0
        assert f1_measure._true_negatives == 2.0
        assert f1_measure._false_positives == 3.0
        assert f1_measure._false_negatives == 0.0
        f1_measure.reset()
        # check value
        assert_allclose(precision, 0.25)
        assert_allclose(recall, 1.0)
        assert_allclose(f1, 0.4)
        # check type
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(f1, float)

    @multi_device
    def test_f1_measure_accumulates_and_resets_correctly(self, device: str):
        f1_measure = F1Measure(positive_label=0)
        predictions = torch.tensor(
            [
                [0.35, 0.25, 0.1, 0.1, 0.2],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.5, 0.1, 0.2, 0.0],
                [0.1, 0.2, 0.1, 0.7, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
            ],
            device=device,
        )
        # [True Positive, True Negative, True Negative,
        #  False Negative, True Negative, False Negative]
        targets = torch.tensor([0, 4, 1, 0, 3, 0], device=device)
        f1_measure(predictions, targets)
        f1_measure(predictions, targets)
        metrics = f1_measure.get_metric()
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        assert f1_measure._true_positives == 2.0
        assert f1_measure._true_negatives == 6.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 4.0
        f1_measure.reset()
        assert_allclose(precision, 1.0)
        assert_allclose(recall, 0.333333333)
        assert_allclose(f1, 0.499999999)
        assert f1_measure._true_positives == 0.0
        assert f1_measure._true_negatives == 0.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 0.0

    @multi_device
    def test_f1_measure_works_for_sequences(self, device: str):
        f1_measure = F1Measure(positive_label=0)
        predictions = torch.tensor(
            [
                [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]],
                [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]],
            ],
            device=device,
        )
        # [[True Positive, True Negative, True Negative],
        #  [True Positive, True Negative, False Negative]]
        targets = torch.tensor([[0, 3, 4], [0, 1, 0]], device=device)
        f1_measure(predictions, targets)
        metrics = f1_measure.get_metric()
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        assert f1_measure._true_positives == 2.0
        assert f1_measure._true_negatives == 3.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 1.0
        f1_measure.reset()
        assert_allclose(precision, 1.0)
        assert_allclose(recall, 0.666666666)
        assert_allclose(f1, 0.8)

        # Test the same thing with a mask:
        mask = torch.tensor([[False, True, False], [True, True, True]], device=device)
        f1_measure(predictions, targets, mask)
        metrics = f1_measure.get_metric()
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        assert f1_measure._true_positives == 1.0
        assert f1_measure._true_negatives == 2.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 1.0
        assert_allclose(precision, 1.0)
        assert_allclose(recall, 0.5)
        assert_allclose(f1, 0.66666666666)

    def test_distributed_fbeta_measure(self):
        predictions = [
            torch.tensor(
                [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]]
            ),
            torch.tensor(
                [[0.1, 0.5, 0.1, 0.2, 0.0], [0.1, 0.2, 0.1, 0.7, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]]
            ),
        ]
        targets = [torch.tensor([0, 4, 1]), torch.tensor([0, 3, 0])]
        metric_kwargs = {"predictions": predictions, "gold_labels": targets}
        desired_metrics = {
            "precision": 1.0,
            "recall": 0.333333333,
            "f1": 0.499999999,
        }
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            F1Measure(positive_label=0),
            metric_kwargs,
            desired_metrics,
            exact=False,
        )
