from typing import List

import numpy
import torch
from allennlp.common.checks import ConfigurationError

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import FBetaMeasure


class FBetaMeasureTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        # [0, 1, 1, 1, 3, 1]
        self.predictions = torch.Tensor(
            [
                [0.35, 0.25, 0.1, 0.1, 0.2],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.5, 0.1, 0.2, 0.0],
                [0.1, 0.2, 0.1, 0.7, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
            ]
        )
        self.targets = torch.Tensor([0, 4, 1, 0, 3, 0])

        # detailed target state
        self.pred_sum = [1, 4, 0, 1, 0]
        self.true_sum = [3, 1, 0, 1, 1]
        self.true_positive_sum = [1, 1, 0, 1, 0]
        self.true_negative_sum = [3, 2, 6, 5, 5]
        self.total_sum = [6, 6, 6, 6, 6]

        desired_precisions = [1.00, 0.25, 0.00, 1.00, 0.00]
        desired_recalls = [0.33, 1.00, 0.00, 1.00, 0.00]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]
        self.desired_precisions = desired_precisions
        self.desired_recalls = desired_recalls
        self.desired_fscores = desired_fscores

    def test_config_errors(self):
        # Bad beta
        self.assertRaises(ConfigurationError, FBetaMeasure, beta=0.0)

        # Bad average option
        self.assertRaises(ConfigurationError, FBetaMeasure, average="mega")

        # Empty input labels
        self.assertRaises(ConfigurationError, FBetaMeasure, labels=[])

    def test_runtime_errors(self):
        fbeta = FBetaMeasure()
        # Metric was never called.
        self.assertRaises(RuntimeError, fbeta.get_metric)

    def test_fbeta_multiclass_state(self):
        fbeta = FBetaMeasure()
        fbeta(self.predictions, self.targets)

        # check state
        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), self.pred_sum)
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), self.true_sum)
        numpy.testing.assert_almost_equal(fbeta._true_positive_sum.tolist(), self.true_positive_sum)
        numpy.testing.assert_almost_equal(fbeta._true_negative_sum.tolist(), self.true_negative_sum)
        numpy.testing.assert_almost_equal(fbeta._total_sum.tolist(), self.total_sum)

    def test_fbeta_multiclass_metric(self):
        fbeta = FBetaMeasure()
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        # check value
        numpy.testing.assert_almost_equal(precisions, self.desired_precisions, decimal=2)
        numpy.testing.assert_almost_equal(recalls, self.desired_recalls, decimal=2)
        numpy.testing.assert_almost_equal(fscores, self.desired_fscores, decimal=2)

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    def test_fbeta_multiclass_with_mask(self):
        mask = torch.Tensor([1, 1, 1, 1, 1, 0])

        fbeta = FBetaMeasure()
        fbeta(self.predictions, self.targets, mask)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), [1, 3, 0, 1, 0])
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), [2, 1, 0, 1, 1])
        numpy.testing.assert_almost_equal(fbeta._true_positive_sum.tolist(), [1, 1, 0, 1, 0])

        desired_precisions = [1.00, 0.33, 0.00, 1.00, 0.00]
        desired_recalls = [0.50, 1.00, 0.00, 1.00, 0.00]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]
        numpy.testing.assert_almost_equal(precisions, desired_precisions, decimal=2)
        numpy.testing.assert_almost_equal(recalls, desired_recalls, decimal=2)
        numpy.testing.assert_almost_equal(fscores, desired_fscores, decimal=2)

    def test_fbeta_multiclass_macro_average_metric(self):
        fbeta = FBetaMeasure(average="macro")
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        macro_precision = numpy.mean(self.desired_precisions)
        macro_recall = numpy.mean(self.desired_recalls)
        macro_fscore = numpy.mean(self.desired_fscores)
        # check value
        numpy.testing.assert_almost_equal(precisions, macro_precision, decimal=2)
        numpy.testing.assert_almost_equal(recalls, macro_recall, decimal=2)
        numpy.testing.assert_almost_equal(fscores, macro_fscore, decimal=2)

        # check type
        assert isinstance(precisions, float)
        assert isinstance(recalls, float)
        assert isinstance(fscores, float)

    def test_fbeta_multiclass_micro_average_metric(self):
        fbeta = FBetaMeasure(average="micro")
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        true_positives = [1, 1, 0, 1, 0]
        false_positives = [0, 3, 0, 0, 0]
        false_negatives = [2, 0, 0, 0, 1]
        mean_true_positive = numpy.mean(true_positives)
        mean_false_positive = numpy.mean(false_positives)
        mean_false_negative = numpy.mean(false_negatives)

        micro_precision = mean_true_positive / (mean_true_positive + mean_false_positive)
        micro_recall = mean_true_positive / (mean_true_positive + mean_false_negative)
        micro_fscore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        # check value
        numpy.testing.assert_almost_equal(precisions, micro_precision, decimal=2)
        numpy.testing.assert_almost_equal(recalls, micro_recall, decimal=2)
        numpy.testing.assert_almost_equal(fscores, micro_fscore, decimal=2)

    def test_fbeta_multiclass_with_explicit_labels(self):
        # same prediction but with and explicit label ordering
        fbeta = FBetaMeasure(labels=[4, 3, 2, 1, 0])
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        desired_precisions = self.desired_precisions[::-1]
        desired_recalls = self.desired_recalls[::-1]
        desired_fscores = self.desired_fscores[::-1]
        # check value
        numpy.testing.assert_almost_equal(precisions, desired_precisions, decimal=2)
        numpy.testing.assert_almost_equal(recalls, desired_recalls, decimal=2)
        numpy.testing.assert_almost_equal(fscores, desired_fscores, decimal=2)

    def test_fbeta_multiclass_with_macro_average(self):
        labels = [0, 1]
        fbeta = FBetaMeasure(average="macro", labels=labels)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        macro_precision = numpy.array(self.desired_precisions)[labels].mean()
        macro_recall = numpy.array(self.desired_recalls)[labels].mean()
        macro_fscore = numpy.array(self.desired_fscores)[labels].mean()

        # check value
        numpy.testing.assert_almost_equal(precisions, macro_precision, decimal=2)
        numpy.testing.assert_almost_equal(recalls, macro_recall, decimal=2)
        numpy.testing.assert_almost_equal(fscores, macro_fscore, decimal=2)

    def test_fbeta_multiclass_with_micro_average(self):
        labels = [1, 3]
        fbeta = FBetaMeasure(average="micro", labels=labels)
        fbeta(self.predictions, self.targets)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]
        fscores = metric["fscore"]

        true_positives = [1, 1]
        false_positives = [3, 0]
        false_negatives = [0, 0]
        mean_true_positive = numpy.mean(true_positives)
        mean_false_positive = numpy.mean(false_positives)
        mean_false_negative = numpy.mean(false_negatives)

        micro_precision = mean_true_positive / (mean_true_positive + mean_false_positive)
        micro_recall = mean_true_positive / (mean_true_positive + mean_false_negative)
        micro_fscore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        # check value
        numpy.testing.assert_almost_equal(precisions, micro_precision, decimal=2)
        numpy.testing.assert_almost_equal(recalls, micro_recall, decimal=2)
        numpy.testing.assert_almost_equal(fscores, micro_fscore, decimal=2)

    def test_fbeta_handles_batch_size_of_one(self):
        predictions = torch.Tensor([[0.2862, 0.3479, 0.1627, 0.2033]])
        targets = torch.Tensor([1])
        mask = torch.Tensor([1])

        fbeta = FBetaMeasure()
        fbeta(predictions, targets, mask)
        metric = fbeta.get_metric()
        precisions = metric["precision"]
        recalls = metric["recall"]

        numpy.testing.assert_almost_equal(precisions, [0.0, 1.0, 0.0, 0.0])
        numpy.testing.assert_almost_equal(recalls, [0.0, 1.0, 0.0, 0.0])
