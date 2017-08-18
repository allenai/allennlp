# pylint: disable=no-self-use,invalid-name,protected-access
import torch
import pytest
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import F1Measure


class F1MeasureTest(AllenNlpTestCase):
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
        assert f1_measure._true_positives == 1.0
        assert f1_measure._true_negatives == 3.
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 2.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.333333333)
        numpy.testing.assert_almost_equal(f1, 0.499999999)

        # Test the same thing with a mask:
        mask = torch.Tensor([1, 0, 1, 1, 1, 0])
        f1_measure(predictions, targets, mask)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure._true_positives == 1.0
        assert f1_measure._true_negatives == 2.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 1.0
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
        assert f1_measure._true_positives == 2.0
        assert f1_measure._true_negatives == 6.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 4.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.333333333)
        numpy.testing.assert_almost_equal(f1, 0.499999999)
        assert f1_measure._true_positives == 0.0
        assert f1_measure._true_negatives == 0.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 0.0

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
        assert f1_measure._true_positives == 2.0
        assert f1_measure._true_negatives == 3.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 1.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.666666666)
        numpy.testing.assert_almost_equal(f1, 0.8)

        # Test the same thing with a mask:
        mask = torch.Tensor([[0, 1, 0],
                             [1, 1, 1]])
        f1_measure(predictions, targets, mask)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure._true_positives == 1.0
        assert f1_measure._true_negatives == 2.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 1.0
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.5)
        numpy.testing.assert_almost_equal(f1, 0.66666666666)
