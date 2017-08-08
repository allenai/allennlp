# pylint: disable=no-self-use,invalid-name
import torch
import pytest
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

class MetricsTest(AllenNlpTestCase):

    def test_categorical_accuracy(self):
        accuracy = CategoricalAccuracy()
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 3])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == {"accuracy": 50.0}

    def test_top_k_categorical_accuracy(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 3])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == {"accuracy": 100.0}

    def test_top_k_categorical_accuracy_works_for_sequences(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.1, 0.6, 0.1, 0.2, 0.0]],
                                    [[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.1, 0.6, 0.1, 0.2, 0.0]]])
        targets = torch.Tensor([[0, 3, 4],
                                [0, 1, 4]])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric()
        numpy.testing.assert_almost_equal(actual_accuracy["accuracy"], 66.6666666)

    def test_top_k_categorical_accuracy_catches_exceptions(self):
        accuracy = CategoricalAccuracy()
        predictions = torch.rand([5, 7])
        out_of_range_labels = torch.Tensor([10, 3, 4, 0, 1])
        with pytest.raises(ConfigurationError):
            accuracy(predictions, out_of_range_labels)

    def test_f1_measure(self):
        f1_measure = F1Measure(null_prediction_label=0)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.5, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.1, 0.7, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])

        # [True Negative, False Negative, True Positive,
        #  True Positive, True Positive, False Negative]
        targets = torch.Tensor([0, 5, 1, 0, 3, 0])
        f1_measure(predictions, targets)
        f1_metrics = f1_measure.get_metric()
        assert f1_measure.true_positives == 2.0
        assert f1_measure.true_negatives == 1.
        assert f1_measure.false_positives == 1.0
        assert f1_measure.false_negatives == 2.0
        numpy.testing.assert_almost_equal(f1_metrics["precision"], 0.666666666)
        numpy.testing.assert_almost_equal(f1_metrics["recall"], 0.5)
        numpy.testing.assert_almost_equal(f1_metrics["f1-measure"], 0.57142857)

    def test_f1_measure_works_for_sequences(self):
        f1_measure = F1Measure(null_prediction_label=0)
        predictions = torch.Tensor([[[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.1, 0.6, 0.1, 0.2, 0.0]],
                                    [[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.1, 0.6, 0.1, 0.2, 0.0]]])

        # [[True Negative, False Positive, False Positive],
        #  [True Negative, True Positive, False Negative]]
        targets = torch.Tensor([[0, 3, 4],
                                [0, 1, 0]])

        f1_measure(predictions, targets)
        f1_metrics = f1_measure.get_metric()
        assert f1_measure.true_positives == 1.0
        assert f1_measure.true_negatives == 2.
        assert f1_measure.false_positives == 2.0
        assert f1_measure.false_negatives == 1.0
        numpy.testing.assert_almost_equal(f1_metrics["precision"], 0.333333333)
        numpy.testing.assert_almost_equal(f1_metrics["recall"], 0.5)
        numpy.testing.assert_almost_equal(f1_metrics["f1-measure"], 0.39999999)
