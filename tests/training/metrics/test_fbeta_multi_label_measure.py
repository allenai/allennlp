from typing import List

import pytest
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.testing import assert_allclose

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    run_distributed_test,
    global_distributed_metric,
)
from allennlp_multi_label import FBetaMeasureMultiLabel


class TestFBetaMeasureMultiLabel(AllenNlpTestCase):
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
        # I don't think this is correct for multi-label
        self.true_negative_sum = [10.0, 10.0, 12.0, 11.0, 14.0]
        self.total_sum = [15, 15, 15, 15, 15]

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
        pytest.raises(ConfigurationError, FBetaMeasureMultiLabel, beta=0.0)

        # Bad average option
        pytest.raises(ConfigurationError, FBetaMeasureMultiLabel, average="mega")

        # Empty input labels
        pytest.raises(ConfigurationError, FBetaMeasureMultiLabel, labels=[])

    @multi_device
    def test_runtime_errors(self, device: str):
        fbeta = FBetaMeasureMultiLabel()
        # Metric was never called.
        pytest.raises(RuntimeError, fbeta.get_metric)

    @multi_device
    def test_fbeta_multilabel_state(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaMeasureMultiLabel()
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

        fbeta = FBetaMeasureMultiLabel()
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
    def test_fbeta_multilabel_with_mask(self, device: str):
        assert False

    @multi_device
    def test_fbeta_multilabel_macro_average_metric(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaMeasureMultiLabel(average="macro")
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

        fbeta = FBetaMeasureMultiLabel(average="micro")
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
        fbeta = FBetaMeasureMultiLabel(labels=[4, 3, 2, 1, 0])
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
        fbeta = FBetaMeasureMultiLabel(average="macro", labels=labels)
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
        fbeta = FBetaMeasureMultiLabel(average="micro", labels=labels)
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
        fbeta = FBetaMeasureMultiLabel(average="weighted", labels=labels)
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
        assert False

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_false_last_class(self, device: str):
        assert False

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_true_last_class(self, device: str):
        assert False

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_true_other_class(self, device: str):
        assert False

    @multi_device
    def test_fbeta_multilabel_handles_no_prediction_true_all_class(self, device: str):
        assert False

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
            FBetaMeasureMultiLabel(),
            metric_kwargs,
            desired_metrics,
            exact=False,
        )
