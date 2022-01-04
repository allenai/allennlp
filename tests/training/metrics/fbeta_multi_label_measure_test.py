from typing import Dict, List, Tuple, Union, Any

import pytest
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import (
    AllenNlpTestCase,
    global_distributed_metric,
    multi_device,
    run_distributed_test,
)
from sklearn.metrics import precision_recall_fscore_support
from torch.testing import assert_allclose

from allennlp.training.metrics import FBetaMultiLabelMeasure


class FBetaMultiLabelMeasureTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.predictions = torch.tensor(
            [
                [0.55, 0.25, 0.10, 0.10, 0.20],
                [0.10, 0.60, 0.10, 0.95, 0.00],
                [0.90, 0.80, 0.75, 0.80, 0.00],
                [0.49, 0.50, 0.95, 0.55, 0.00],
                [0.60, 0.49, 0.60, 0.65, 0.85],
                [0.85, 0.40, 0.10, 0.20, 0.00],
            ]
        )
        self.targets = torch.tensor(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        # detailed target state
        self.pred_sum = [4, 3, 3, 4, 1]
        self.true_sum = [4, 5, 2, 4, 0]
        self.true_positive_sum = [3, 3, 2, 4, 0]
        self.true_negative_sum = [1, 1, 3, 2, 5]
        self.total_sum = [30, 30, 30, 30, 30]

        # true_positive_sum / pred_sum
        desired_precisions = [3 / 4, 3 / 3, 2 / 3, 4 / 4, 0 / 1]
        # true_positive_sum / true_sum
        desired_recalls = [3 / 4, 3 / 5, 2 / 2, 4 / 4, 0.00]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]
        self.desired_precisions = desired_precisions
        self.desired_recalls = desired_recalls
        self.desired_fscores = desired_fscores

    @multi_device
    def test_config_errors(self, device: str):
        # Bad beta
        pytest.raises(ConfigurationError, FBetaMultiLabelMeasure, beta=0.0)

        # Bad average option
        pytest.raises(ConfigurationError, FBetaMultiLabelMeasure, average="mega")

        # Empty input labels
        pytest.raises(ConfigurationError, FBetaMultiLabelMeasure, labels=[])

    @multi_device
    def test_runtime_errors(self, device: str):
        fbeta = FBetaMultiLabelMeasure()
        # Metric was never called.
        pytest.raises(RuntimeError, fbeta.get_metric)

    @multi_device
    def test_fbeta_multilabel_state(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaMultiLabelMeasure()
        fbeta(self.predictions, self.targets)

        # check state
        assert_allclose(fbeta._pred_sum.tolist(), self.pred_sum)
        assert_allclose(fbeta._true_sum.tolist(), self.true_sum)
        assert_allclose(fbeta._true_positive_sum.tolist(), self.true_positive_sum)
        assert_allclose(fbeta._true_negative_sum.tolist(), self.true_negative_sum)
        assert_allclose(fbeta._total_sum.tolist(), self.total_sum)

    @multi_device
    def test_fbeta_multilabel_metric(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaMultiLabelMeasure()
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # check value
        assert_allclose(precisions, self.desired_precisions)
        assert_allclose(recalls, self.desired_recalls)
        assert_allclose(fscores, self.desired_fscores)

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    @multi_device
    def test_fbeta_multilable_with_extra_dimensions(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaMultiLabelMeasure()
        fbeta(self.predictions.unsqueeze(1), self.targets.unsqueeze(1))
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # check value
        assert_allclose(precisions, self.desired_precisions)
        assert_allclose(recalls, self.desired_recalls)
        assert_allclose(fscores, self.desired_fscores)

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    @multi_device
    def test_fbeta_multilabel_with_mask(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        mask = torch.tensor([True, True, True, True, True, False], device=device).unsqueeze(-1)

        fbeta = FBetaMultiLabelMeasure()
        fbeta(self.predictions, self.targets, mask)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        assert_allclose(fbeta._pred_sum.tolist(), [3, 3, 3, 4, 1])
        assert_allclose(fbeta._true_sum.tolist(), [4, 5, 2, 4, 0])
        assert_allclose(fbeta._true_positive_sum.tolist(), [3, 3, 2, 4, 0])

        desired_precisions = [3 / 3, 3 / 3, 2 / 3, 4 / 4, 0 / 1]
        desired_recalls = [3 / 4, 3 / 5, 2 / 2, 4 / 4, 0.00]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

    @multi_device
    def test_fbeta_multilabel_macro_average_metric(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaMultiLabelMeasure(average="macro")
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # We keep the expected values in CPU because FBetaMeasure returns them in CPU.
        macro_precision = torch.tensor(self.desired_precisions).mean()
        macro_recall = torch.tensor(self.desired_recalls).mean()
        macro_fscore = torch.tensor(self.desired_fscores).mean()
        # check value
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)

        # check type
        assert isinstance(precisions, float)
        assert isinstance(recalls, float)
        assert isinstance(fscores, float)

    @multi_device
    def test_fbeta_multilabel_micro_average_metric(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaMultiLabelMeasure(average="micro")
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # We keep the expected values in CPU because FBetaMeasure returns them in CPU.
        true_positives = torch.tensor([3, 3, 2, 4, 0], dtype=torch.float32)
        false_positives = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)
        false_negatives = torch.tensor([1, 2, 0, 0, 0], dtype=torch.float32)
        mean_true_positive = true_positives.mean()
        mean_false_positive = false_positives.mean()
        mean_false_negative = false_negatives.mean()

        micro_precision = mean_true_positive / (mean_true_positive + mean_false_positive)
        micro_recall = mean_true_positive / (mean_true_positive + mean_false_negative)
        micro_fscore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        # check value
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)

    @multi_device
    def test_fbeta_multilabel_with_explicit_labels(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        # same prediction but with and explicit label ordering
        fbeta = FBetaMultiLabelMeasure(labels=[4, 3, 2, 1, 0])
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        desired_precisions = self.desired_precisions[::-1]
        desired_recalls = self.desired_recalls[::-1]
        desired_fscores = self.desired_fscores[::-1]
        # check value
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

    @multi_device
    def test_fbeta_multilabel_with_macro_average(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        labels = [0, 1]
        fbeta = FBetaMultiLabelMeasure(average="macro", labels=labels)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # We keep the expected values in CPU because FBetaMeasure returns them in CPU.
        macro_precision = torch.tensor(self.desired_precisions)[labels].mean()
        macro_recall = torch.tensor(self.desired_recalls)[labels].mean()
        macro_fscore = torch.tensor(self.desired_fscores)[labels].mean()

        # check value
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)

    @multi_device
    def test_fbeta_multilabel_with_micro_average(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        labels = [1, 3]
        fbeta = FBetaMultiLabelMeasure(average="micro", labels=labels)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # We keep the expected values in CPU because FBetaMeasure returns them in CPU.
        true_positives = torch.tensor([3, 4], dtype=torch.float32)
        false_positives = torch.tensor([0, 0], dtype=torch.float32)
        false_negatives = torch.tensor([2, 0], dtype=torch.float32)
        mean_true_positive = true_positives.mean()
        mean_false_positive = false_positives.mean()
        mean_false_negative = false_negatives.mean()

        micro_precision = mean_true_positive / (mean_true_positive + mean_false_positive)
        micro_recall = mean_true_positive / (mean_true_positive + mean_false_negative)
        micro_fscore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        # check value
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)

    @multi_device
    def test_fbeta_multilabel_with_weighted_average(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        labels = [0, 1]
        fbeta = FBetaMultiLabelMeasure(average="weighted", labels=labels)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        weighted_precision, weighted_recall, weighted_fscore, _ = precision_recall_fscore_support(
            self.targets.cpu().numpy(),
            torch.where(
                self.predictions >= fbeta._threshold,
                torch.ones_like(self.predictions),
                torch.zeros_like(self.predictions),
            )
            .cpu()
            .numpy(),
            labels=labels,
            average="weighted",
        )

        # check value
        assert_allclose(precisions, weighted_precision)
        assert_allclose(recalls, weighted_recall)
        assert_allclose(fscores, weighted_fscore)

    @multi_device
    def test_fbeta_multilabel_handles_batch_size_of_one(self, device: str):
        predictions = torch.tensor([[0.2862, 0.5479, 0.1627, 0.2033]], device=device)
        targets = torch.tensor([[0, 1, 0, 0]], device=device)
        mask = torch.tensor([[True]], device=device)

        fbeta = FBetaMultiLabelMeasure()
        fbeta(predictions, targets, mask)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]

        assert_allclose(precisions, [0.0, 1.0, 0.0, 0.0])
        assert_allclose(recalls, [0.0, 1.0, 0.0, 0.0])

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_false_last_class(self, device: str):

        predictions = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        # preds = [0, NA]
        targets = torch.tensor([[1, 0], [1, 0]], device=device)

        fbeta = FBetaMultiLabelMeasure()
        fbeta(predictions, targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        assert_allclose(precisions, [1.0, 0.0])
        assert_allclose(recalls, [0.5, 0.0])
        assert_allclose(fscores, [0.6667, 0.0])

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_true_last_class(self, device: str):

        predictions = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        # preds = [0, NA]
        targets = torch.tensor([[1, 0], [0, 1]], device=device)

        fbeta = FBetaMultiLabelMeasure()
        fbeta(predictions, targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        assert_allclose(precisions, [1.0, 0.0])
        assert_allclose(recalls, [1.0, 0.0])
        assert_allclose(fscores, [1.0, 0.0])

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_true_other_class(self, device: str):
        predictions = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        # preds = [0, NA]
        targets = torch.tensor([[0, 1], [1, 0]], device=device)

        fbeta = FBetaMultiLabelMeasure()
        fbeta(predictions, targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        assert_allclose(precisions, [0.0, 0.0])
        assert_allclose(recalls, [0.0, 0.0])
        assert_allclose(fscores, [0.0, 0.0])

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_true_all_class(self, device: str):
        predictions = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        # preds = [0, NA]
        targets = torch.tensor([[0, 1], [0, 1]], device=device)

        fbeta = FBetaMultiLabelMeasure()
        fbeta(predictions, targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        assert_allclose(precisions, [0.0, 0.0])
        assert_allclose(recalls, [0.0, 0.0])
        assert_allclose(fscores, [0.0, 0.0])

    def test_distributed_fbeta_multilabel_measure(self):
        predictions = [
            torch.tensor(
                [
                    [0.55, 0.25, 0.10, 0.10, 0.20],
                    [0.10, 0.60, 0.10, 0.95, 0.00],
                    [0.90, 0.80, 0.75, 0.80, 0.00],
                ]
            ),
            torch.tensor(
                [
                    [0.49, 0.50, 0.95, 0.55, 0.00],
                    [0.60, 0.49, 0.60, 0.65, 0.85],
                    [0.85, 0.40, 0.10, 0.20, 0.00],
                ]
            ),
        ]

        targets = [
            torch.tensor([[1, 1, 0, 0, 0], [0, 1, 0, 1, 0], [1, 1, 0, 1, 0]]),
            torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]]),
        ]

        metric_kwargs = {"predictions": predictions, "gold_labels": targets}
        desired_metrics = {
            "precision": self.desired_precisions,
            "recall": self.desired_recalls,
            "fscore": self.desired_fscores,
        }
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            FBetaMultiLabelMeasure(),
            metric_kwargs,
            desired_metrics,
            exact=False,
        )

    def test_multiple_distributed_runs(self):
        predictions = [
            torch.tensor(
                [
                    [0.55, 0.25, 0.10, 0.10, 0.20],
                    [0.10, 0.60, 0.10, 0.95, 0.00],
                    [0.90, 0.80, 0.75, 0.80, 0.00],
                ]
            ),
            torch.tensor(
                [
                    [0.49, 0.50, 0.95, 0.55, 0.00],
                    [0.60, 0.49, 0.60, 0.65, 0.85],
                    [0.85, 0.40, 0.10, 0.20, 0.00],
                ]
            ),
        ]
        targets = [
            torch.tensor([[1, 1, 0, 0, 0], [0, 1, 0, 1, 0], [1, 1, 0, 1, 0]]),
            torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]]),
        ]
        metric_kwargs = {"predictions": predictions, "gold_labels": targets}
        desired_metrics = {
            "precision": self.desired_precisions,
            "recall": self.desired_recalls,
            "fscore": self.desired_fscores,
        }
        run_distributed_test(
            [-1, -1],
            multiple_runs,
            FBetaMultiLabelMeasure(),
            metric_kwargs,
            desired_metrics,
            exact=False,
        )


def multiple_runs(
    global_rank: int,
    world_size: int,
    gpu_id: Union[int, torch.device],
    metric: FBetaMultiLabelMeasure,
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

    metric_values = metric.get_metric()

    for key in desired_values:
        assert_allclose(desired_values[key], metric_values[key])
