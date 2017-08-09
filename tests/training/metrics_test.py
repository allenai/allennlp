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
        assert actual_accuracy == 0.50

    def test_top_k_categorical_accuracy(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 3])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 1.0

    def test_top_k_categorical_accuracy_accumulates_and_resets_correctly(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 3])
        accuracy(predictions, targets)
        accuracy(predictions, targets)
        accuracy(predictions, torch.Tensor([4, 4]))
        accuracy(predictions, torch.Tensor([4, 4]))
        actual_accuracy = accuracy.get_metric(reset=True)
        assert actual_accuracy == 0.50
        assert accuracy.correct_count == 0.0
        assert accuracy.total_count == 0.0

    def test_top_k_categorical_accuracy_respects_mask(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.5, 0.2, 0.0]])
        targets = torch.Tensor([0, 3, 0])
        mask = torch.Tensor([0, 1, 1])
        accuracy(predictions, targets, mask)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 0.50

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
        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 0.6666666)

        # Test the same thing but with a mask:
        mask = torch.Tensor([[0, 1, 1],
                             [1, 0, 1]])
        accuracy(predictions, targets, mask)
        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 0.50)

    def test_top_k_categorical_accuracy_catches_exceptions(self):
        accuracy = CategoricalAccuracy()
        predictions = torch.rand([5, 7])
        out_of_range_labels = torch.Tensor([10, 3, 4, 0, 1])
        with pytest.raises(ConfigurationError):
            accuracy(predictions, out_of_range_labels)

    def test__f1_measure_catches_exceptions(self):
        f1_measure = F1Measure(0)
        predictions = torch.rand([5, 7])
        out_of_range_labels = torch.Tensor([10, 3, 4, 0, 1])
        with pytest.raises(ConfigurationError):
            f1_measure(predictions, out_of_range_labels)

    def test_f1_measure(self):
        f1_measure = F1Measure(positive_label=0)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.5, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.1, 0.7, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        # [True Positive, True Negative, True Negative,
        #  False Negative, True Negative, False Negative]
        targets = torch.Tensor([0, 4, 1, 0, 3, 0])
        f1_measure(predictions, targets)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure.true_positives == 1.0
        assert f1_measure.true_negatives == 3.
        assert f1_measure.false_positives == 0.0
        assert f1_measure.false_negatives == 2.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.333333333)
        numpy.testing.assert_almost_equal(f1, 0.499999999)

        # Test the same thing with a mask:
        mask = torch.Tensor([1, 0, 1, 1, 1, 0])
        f1_measure(predictions, targets, mask)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure.true_positives == 1.0
        assert f1_measure.true_negatives == 2.0
        assert f1_measure.false_positives == 0.0
        assert f1_measure.false_negatives == 1.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.5)
        numpy.testing.assert_almost_equal(f1, 0.6666666666)

    def test_f1_measure_accumulates_and_resets_correctly(self):
        f1_measure = F1Measure(positive_label=0)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.5, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.1, 0.7, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        # [True Positive, True Negative, True Negative,
        #  False Negative, True Negative, False Negative]
        targets = torch.Tensor([0, 4, 1, 0, 3, 0])
        f1_measure(predictions, targets)
        f1_measure(predictions, targets)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure.true_positives == 2.0
        assert f1_measure.true_negatives == 6.0
        assert f1_measure.false_positives == 0.0
        assert f1_measure.false_negatives == 4.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.333333333)
        numpy.testing.assert_almost_equal(f1, 0.499999999)
        assert f1_measure.true_positives == 0.0
        assert f1_measure.true_negatives == 0.0
        assert f1_measure.false_positives == 0.0
        assert f1_measure.false_negatives == 0.0

    def test_f1_measure_works_for_sequences(self):
        f1_measure = F1Measure(positive_label=0)
        predictions = torch.Tensor([[[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.1, 0.6, 0.1, 0.2, 0.0]],
                                    [[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.1, 0.6, 0.1, 0.2, 0.0]]])
        # [[True Positive, True Negative, True Negative],
        #  [True Positive, True Negative, False Negative]]
        targets = torch.Tensor([[0, 3, 4],
                                [0, 1, 0]])
        f1_measure(predictions, targets)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure.true_positives == 2.0
        assert f1_measure.true_negatives == 3.0
        assert f1_measure.false_positives == 0.0
        assert f1_measure.false_negatives == 1.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.666666666)
        numpy.testing.assert_almost_equal(f1, 0.8)

        # Test the same thing with a mask:
        mask = torch.Tensor([[0, 1, 0],
                             [1, 1, 1]])
        f1_measure(predictions, targets, mask)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure.true_positives == 1.0
        assert f1_measure.true_negatives == 2.0
        assert f1_measure.false_positives == 0.0
        assert f1_measure.false_negatives == 1.0
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.5)
        numpy.testing.assert_almost_equal(f1, 0.66666666666)
