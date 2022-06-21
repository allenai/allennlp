from typing import Any, Dict, List, Tuple, Union

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch.testing import assert_allclose
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    run_distributed_test,
    global_distributed_metric,
)
from allennlp.training.metrics import FBetaVerboseMeasure


class FBetaVerboseMeasureTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        # [0, 1, 1, 1, 3, 1]
        self.predictions = torch.tensor(
            [
                [0.35, 0.25, 0.1, 0.1, 0.2],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.5, 0.1, 0.2, 0.0],
                [0.1, 0.2, 0.1, 0.7, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
            ]
        )
        self.targets = torch.tensor([0, 4, 1, 0, 3, 0])

        # detailed target state
        self.pred_sum = [1, 4, 0, 1, 0]
        self.true_sum = [3, 1, 0, 1, 1]
        self.true_positive_sum = [1, 1, 0, 1, 0]
        self.true_negative_sum = [3, 2, 6, 5, 5]
        self.total_sum = [6, 6, 6, 6, 6]

        desired_precisions = [1.00, 0.25, 0.00, 1.00, 0.00]
        desired_recalls = [1 / 3, 1.00, 0.00, 1.00, 0.00]
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
        pytest.raises(ConfigurationError, FBetaVerboseMeasure, beta=0.0)

        # Empty input labels
        pytest.raises(ConfigurationError, FBetaVerboseMeasure, labels=[])

    @multi_device
    def test_runtime_errors(self, device: str):
        fbeta = FBetaVerboseMeasure()
        # Metric was never called.
        pytest.raises(RuntimeError, fbeta.get_metric)

    @multi_device
    def test_fbeta_multiclass_state(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets)

        # check state
        assert_allclose(fbeta._pred_sum.tolist(), self.pred_sum)
        assert_allclose(fbeta._true_sum.tolist(), self.true_sum)
        assert_allclose(fbeta._true_positive_sum.tolist(), self.true_positive_sum)
        assert_allclose(fbeta._true_negative_sum.tolist(), self.true_negative_sum)
        assert_allclose(fbeta._total_sum.tolist(), self.total_sum)

    @multi_device
    def test_fbeta_multiclass_metric(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = [metric[f"{i}-precision"] for i in range(self.predictions.size(1))]
        recalls = [metric[f"{i}-recall"] for i in range(self.predictions.size(1))]
        fscores = [metric[f"{i}-fscore"] for i in range(self.predictions.size(1))]

        # check value
        assert_allclose(precisions, self.desired_precisions)
        assert_allclose(recalls, self.desired_recalls)
        assert_allclose(fscores, self.desired_fscores)

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    @multi_device
    def test_fbeta_multiclass_with_mask(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        mask = torch.tensor([True, True, True, True, True, False], device=device)

        fbeta = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets, mask)
        metric = fbeta.get_metric()
        precisions = [metric[f"{i}-precision"] for i in range(self.predictions.size(1))]
        recalls = [metric[f"{i}-recall"] for i in range(self.predictions.size(1))]
        fscores = [metric[f"{i}-fscore"] for i in range(self.predictions.size(1))]

        assert_allclose(fbeta._pred_sum.tolist(), [1, 3, 0, 1, 0])
        assert_allclose(fbeta._true_sum.tolist(), [2, 1, 0, 1, 1])
        assert_allclose(fbeta._true_positive_sum.tolist(), [1, 1, 0, 1, 0])

        desired_precisions = [1.00, 1 / 3, 0.00, 1.00, 0.00]
        desired_recalls = [0.50, 1.00, 0.00, 1.00, 0.00]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

    @multi_device
    def test_fbeta_multiclass_macro_average_metric(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["macro-precision"]
        recalls = metric["macro-recall"]
        fscores = metric["macro-fscore"]

        # compute desired values with sklearn
        num_labels = self.predictions.size(1)
        labels = np.arange(num_labels)
        macro_precision, macro_recall, macro_fscore, _ = precision_recall_fscore_support(
            self.targets.cpu().numpy(),
            self.predictions.argmax(dim=1).cpu().numpy(),
            average="macro",
            labels=labels,  # necessary to consider missing labels from 'true' and 'pred'
        )

        # check values
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)

        # check type
        assert isinstance(precisions, float)
        assert isinstance(recalls, float)
        assert isinstance(fscores, float)

    @multi_device
    def test_fbeta_multiclass_micro_average_metric(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["micro-precision"]
        recalls = metric["micro-recall"]
        fscores = metric["micro-fscore"]

        # compute desired values with sklearn
        micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(
            self.targets.cpu().numpy(),
            self.predictions.argmax(dim=1).cpu().numpy(),
            average="micro",
        )

        # check value
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)

    @multi_device
    def test_fbeta_multiclass_weighted_average_metric(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        fbeta = FBetaVerboseMeasure()
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["weighted-precision"]
        recalls = metric["weighted-recall"]
        fscores = metric["weighted-fscore"]

        # compute desired values with sklearn
        weighted_precision, weighted_recall, weighted_fscore, _ = precision_recall_fscore_support(
            self.targets.cpu().numpy(),
            self.predictions.argmax(dim=1).cpu().numpy(),
            average="weighted",
        )

        # check value
        assert_allclose(precisions, weighted_precision)
        assert_allclose(recalls, weighted_recall)
        assert_allclose(fscores, weighted_fscore)

    @multi_device
    def test_fbeta_multiclass_with_explicit_labels(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        # same prediction but with and explicit label ordering
        fbeta = FBetaVerboseMeasure(labels=[4, 3, 2, 1, 0])
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = [metric[f"{i}-precision"] for i in range(self.predictions.size(1))]
        recalls = [metric[f"{i}-recall"] for i in range(self.predictions.size(1))]
        fscores = [metric[f"{i}-fscore"] for i in range(self.predictions.size(1))]

        desired_precisions = self.desired_precisions[::-1]
        desired_recalls = self.desired_recalls[::-1]
        desired_fscores = self.desired_fscores[::-1]

        # check value
        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

    @multi_device
    def test_fbeta_multiclass_with_explicit_labels_macro(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        labels = [0, 1]
        fbeta = FBetaVerboseMeasure(labels=labels)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["macro-precision"]
        recalls = metric["macro-recall"]
        fscores = metric["macro-fscore"]

        # compute desired values with sklearn
        macro_precision, macro_recall, macro_fscore, _ = precision_recall_fscore_support(
            self.targets.cpu().numpy(),
            self.predictions.argmax(dim=1).cpu().numpy(),
            average="macro",
            labels=labels,
        )

        # check value
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)

    @multi_device
    def test_fbeta_multiclass_with_explicit_labels_micro(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        labels = [1, 3]
        fbeta = FBetaVerboseMeasure(labels=labels)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["micro-precision"]
        recalls = metric["micro-recall"]
        fscores = metric["micro-fscore"]

        # compute desired values with sklearn
        micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(
            self.targets.cpu().numpy(),
            self.predictions.argmax(dim=1).cpu().numpy(),
            average="micro",
            labels=labels,
        )

        # check value
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)

    @multi_device
    def test_fbeta_multiclass_with_explicit_labels_weighted(self, device: str):
        self.predictions = self.predictions.to(device)
        self.targets = self.targets.to(device)

        labels = [0, 1]
        fbeta = FBetaVerboseMeasure(labels=labels)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["weighted-precision"]
        recalls = metric["weighted-recall"]
        fscores = metric["weighted-fscore"]

        # compute desired values with sklearn
        weighted_precision, weighted_recall, weighted_fscore, _ = precision_recall_fscore_support(
            self.targets.cpu().numpy(),
            self.predictions.argmax(dim=1).cpu().numpy(),
            labels=labels,
            average="weighted",
        )

        # check value
        assert_allclose(precisions, weighted_precision)
        assert_allclose(recalls, weighted_recall)
        assert_allclose(fscores, weighted_fscore)

    @multi_device
    def test_fbeta_handles_batch_size_of_one(self, device: str):
        predictions = torch.tensor([[0.2862, 0.3479, 0.1627, 0.2033]], device=device)
        targets = torch.tensor([1], device=device)
        mask = torch.tensor([True], device=device)

        fbeta = FBetaVerboseMeasure()
        fbeta(predictions, targets, mask)
        metric = fbeta.get_metric()
        precisions = [metric[f"{i}-precision"] for i in range(predictions.size(1))]
        recalls = [metric[f"{i}-recall"] for i in range(predictions.size(1))]

        assert_allclose(precisions, [0.0, 1.0, 0.0, 0.0])
        assert_allclose(recalls, [0.0, 1.0, 0.0, 0.0])

    @multi_device
    def test_fbeta_handles_no_prediction_false_last_class(self, device: str):

        predictions = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        # preds = [0, NA]
        targets = torch.tensor([0, 0], device=device)

        fbeta = FBetaVerboseMeasure()
        fbeta(predictions, targets)
        metric = fbeta.get_metric()
        precisions = [metric[f"{i}-precision"] for i in range(predictions.size(1))]
        recalls = [metric[f"{i}-recall"] for i in range(predictions.size(1))]
        fscores = [metric[f"{i}-fscore"] for i in range(predictions.size(1))]

        assert_allclose(precisions, [1.0, 0.0])
        assert_allclose(recalls, [0.5, 0.0])
        assert_allclose(fscores, [0.6667, 0.0])

    @multi_device
    def test_fbeta_handles_no_prediction_true_last_class(self, device: str):

        predictions = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        # preds = [0, NA]
        targets = torch.tensor([0, 1], device=device)

        fbeta = FBetaVerboseMeasure()
        fbeta(predictions, targets)
        metric = fbeta.get_metric()
        precisions = [metric[f"{i}-precision"] for i in range(predictions.size(1))]
        recalls = [metric[f"{i}-recall"] for i in range(predictions.size(1))]
        fscores = [metric[f"{i}-fscore"] for i in range(predictions.size(1))]

        assert_allclose(precisions, [1.0, 0.0])
        assert_allclose(recalls, [1.0, 0.0])
        assert_allclose(fscores, [1.0, 0.0])

    @multi_device
    def test_fbeta_handles_no_prediction_true_other_class(self, device: str):

        predictions = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        # preds = [0, NA]
        targets = torch.tensor([1, 0], device=device)

        fbeta = FBetaVerboseMeasure()
        fbeta(predictions, targets)
        metric = fbeta.get_metric()
        precisions = [metric[f"{i}-precision"] for i in range(predictions.size(1))]
        recalls = [metric[f"{i}-recall"] for i in range(predictions.size(1))]
        fscores = [metric[f"{i}-fscore"] for i in range(predictions.size(1))]

        assert_allclose(precisions, [0.0, 0.0])
        assert_allclose(recalls, [0.0, 0.0])
        assert_allclose(fscores, [0.0, 0.0])

    @multi_device
    def test_fbeta_handles_no_prediction_true_all_class(self, device: str):

        predictions = torch.tensor([[0.65, 0.35], [0.0, 0.0]], device=device)
        # preds = [0, NA]
        targets = torch.tensor([1, 1], device=device)

        fbeta = FBetaVerboseMeasure()
        fbeta(predictions, targets)
        metric = fbeta.get_metric()
        precisions = [metric[f"{i}-precision"] for i in range(predictions.size(1))]
        recalls = [metric[f"{i}-recall"] for i in range(predictions.size(1))]
        fscores = [metric[f"{i}-fscore"] for i in range(predictions.size(1))]

        assert_allclose(precisions, [0.0, 0.0])
        assert_allclose(recalls, [0.0, 0.0])
        assert_allclose(fscores, [0.0, 0.0])

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
        desired_metrics = {}
        for i, (p, r, f) in enumerate(
            zip(self.desired_precisions, self.desired_recalls, self.desired_fscores)
        ):
            desired_metrics[f"{i}-precision"] = p
            desired_metrics[f"{i}-recall"] = r
            desired_metrics[f"{i}-fscore"] = f

        # compute desired averages with sklearn
        num_labels = self.predictions.size(1)
        labels = np.arange(num_labels)
        for avg in ["macro", "micro", "weighted"]:
            avg_precision, avg_recall, avg_fscore, _ = precision_recall_fscore_support(
                self.targets.cpu().numpy(),
                self.predictions.argmax(dim=1).cpu().numpy(),
                average=avg,
                labels=labels,  # necessary to consider missing labels from 'true' and 'pred'
            )
            desired_metrics[f"{avg}-precision"] = avg_precision
            desired_metrics[f"{avg}-recall"] = avg_recall
            desired_metrics[f"{avg}-fscore"] = avg_fscore

        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            FBetaVerboseMeasure(),
            metric_kwargs,
            desired_metrics,
            exact=False,
        )

    def test_multiple_distributed_runs(self):
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
        desired_metrics = {}
        for i, (p, r, f) in enumerate(
            zip(self.desired_precisions, self.desired_recalls, self.desired_fscores)
        ):
            desired_metrics[f"{i}-precision"] = p
            desired_metrics[f"{i}-recall"] = r
            desired_metrics[f"{i}-fscore"] = f

        # compute desired averages with sklearn
        num_labels = self.predictions.size(1)
        labels = np.arange(num_labels)
        for avg in ["macro", "micro", "weighted"]:
            avg_precision, avg_recall, avg_fscore, _ = precision_recall_fscore_support(
                self.targets.cpu().numpy(),
                self.predictions.argmax(dim=1).cpu().numpy(),
                average=avg,
                labels=labels,  # necessary to consider missing labels from 'true' and 'pred'
            )
            desired_metrics[f"{avg}-precision"] = avg_precision
            desired_metrics[f"{avg}-recall"] = avg_recall
            desired_metrics[f"{avg}-fscore"] = avg_fscore

        run_distributed_test(
            [-1, -1],
            multiple_runs,
            FBetaVerboseMeasure(),
            metric_kwargs,
            desired_metrics,
            exact=False,
        )


def multiple_runs(
    global_rank: int,
    world_size: int,
    gpu_id: Union[int, torch.device],
    metric: FBetaVerboseMeasure,
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
