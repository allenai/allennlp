import torch
import pytest
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import CategoricalAccuracy


class CategoricalAccuracyTest(AllenNlpTestCase):
    def test_categorical_accuracy(self):
        accuracy = CategoricalAccuracy()
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 3])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 0.50

    def test_top_k_categorical_accuracy(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 3])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 1.0

    def test_top_k_categorical_accuracy_accumulates_and_resets_correctly(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0]])
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
        predictions = torch.Tensor(
            [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.2, 0.5, 0.2, 0.0]]
        )
        targets = torch.Tensor([0, 3, 0])
        mask = torch.Tensor([0, 1, 1])
        accuracy(predictions, targets, mask)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 0.50

    def test_top_k_categorical_accuracy_works_for_sequences(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor(
            [
                [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]],
                [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]],
            ]
        )
        targets = torch.Tensor([[0, 3, 4], [0, 1, 4]])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 0.6666666)

        # Test the same thing but with a mask:
        mask = torch.Tensor([[0, 1, 1], [1, 0, 1]])
        accuracy(predictions, targets, mask)
        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 0.50)

    def test_top_k_categorical_accuracy_catches_exceptions(self):
        accuracy = CategoricalAccuracy()
        predictions = torch.rand([5, 7])
        out_of_range_labels = torch.Tensor([10, 3, 4, 0, 1])
        with pytest.raises(ConfigurationError):
            accuracy(predictions, out_of_range_labels)

    def test_tie_break_categorical_accuracy(self):
        accuracy = CategoricalAccuracy(tie_break=True)
        predictions = torch.Tensor(
            [[0.35, 0.25, 0.35, 0.35, 0.35], [0.1, 0.6, 0.1, 0.2, 0.2], [0.1, 0.0, 0.1, 0.2, 0.2]]
        )
        # Test without mask:
        targets = torch.Tensor([2, 1, 4])
        accuracy(predictions, targets)
        assert accuracy.get_metric(reset=True) == (0.25 + 1 + 0.5) / 3.0

        # # # Test with mask
        mask = torch.Tensor([1, 0, 1])
        targets = torch.Tensor([2, 1, 4])
        accuracy(predictions, targets, mask)
        assert accuracy.get_metric(reset=True) == (0.25 + 0.5) / 2.0

        # # Test tie-break with sequence
        predictions = torch.Tensor(
            [
                [
                    [0.35, 0.25, 0.35, 0.35, 0.35],
                    [0.1, 0.6, 0.1, 0.2, 0.2],
                    [0.1, 0.0, 0.1, 0.2, 0.2],
                ],
                [
                    [0.35, 0.25, 0.35, 0.35, 0.35],
                    [0.1, 0.6, 0.1, 0.2, 0.2],
                    [0.1, 0.0, 0.1, 0.2, 0.2],
                ],
            ]
        )
        targets = torch.Tensor([[0, 1, 3], [0, 3, 4]])  # 0.25 + 1 + 0.5  # 0.25 + 0 + 0.5 = 2.5
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 2.5 / 6.0)

    def test_top_k_and_tie_break_together_catches_exceptions(self):
        with pytest.raises(ConfigurationError):
            CategoricalAccuracy(top_k=2, tie_break=True)

    def test_incorrect_top_k_catches_exceptions(self):
        with pytest.raises(ConfigurationError):
            CategoricalAccuracy(top_k=0)

    def test_does_not_divide_by_zero_with_no_count(self):
        accuracy = CategoricalAccuracy()
        self.assertAlmostEqual(accuracy.get_metric(), 0.0)
