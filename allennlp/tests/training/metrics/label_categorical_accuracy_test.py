# pylint: disable=no-self-use,invalid-name,protected-access
import torch
import pytest
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import LabelCategoricalAccuracy

class LabelCategoricalAccuracyTest(AllenNlpTestCase):
    def test_label_categorical_accuracy(self):
        label_accuracy = LabelCategoricalAccuracy(positive_label=3)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.3, 0.2, 0.1, 0.3, 0.1],
                                    [0.1, 0.05, 0.5, 0.3, 0.05],
                                    [0.1, 0.2, 0.1, 0.6, 0.0]])
        targets = torch.Tensor([0, 3, 2, 1, 3])
        label_accuracy(predictions, targets)
        actual_accuracy = label_accuracy.get_metric()
        assert actual_accuracy == 0.50

    def test_top_k_label_categorical_accuracy(self):
        label_accuracy = LabelCategoricalAccuracy(positive_label=3,
                                                  top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.3, 0.2, 0.1, 0.3, 0.1],
                                    [0.1, 0.05, 0.5, 0.3, 0.05],
                                    [0.1, 0.2, 0.1, 0.6, 0.0]])
        targets = torch.Tensor([0, 3, 2, 1, 3])
        label_accuracy(predictions, targets)
        actual_accuracy = label_accuracy.get_metric()
        assert actual_accuracy == 1.0

    def test_top_k_label_categorical_accuracy_accumulates_and_resets_correctly(self):
        accuracy = LabelCategoricalAccuracy(positive_label=3,
                                            top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.3, 0.2, 0.1, 0.3, 0.1],
                                    [0.1, 0.05, 0.5, 0.3, 0.05],
                                    [0.1, 0.2, 0.1, 0.6, 0.0]])
        targets = torch.Tensor([0, 3, 2, 1, 3])
        accuracy(predictions, targets)
        accuracy(predictions, targets)
        accuracy(predictions, torch.Tensor([3, 4, 2, 3, 2]))
        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 5 / 6)
        assert accuracy.correct_count == 0.0
        assert accuracy.total_count == 0.0

    def test_top_k_label_categorical_accuracy_respects_mask(self):
        accuracy = LabelCategoricalAccuracy(positive_label=3,
                                            top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.3, 0.2, 0.1, 0.3, 0.1],
                                    [0.1, 0.05, 0.5, 0.3, 0.05],
                                    [0.1, 0.2, 0.1, 0.6, 0.0]])
        targets = torch.Tensor([0, 3, 2, 1, 3])
        mask = torch.Tensor([0, 1, 0, 0, 0])
        accuracy(predictions, targets, mask)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 1.0

    def test_top_k_label_categorical_accuracy_works_for_sequences(self):
        accuracy = LabelCategoricalAccuracy(positive_label=3,
                                            top_k=2)
        predictions = torch.Tensor([[[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.3, 0.2, 0.1, 0.3, 0.1],
                                     [0.1, 0.05, 0.5, 0.3, 0.05],
                                     [0.1, 0.2, 0.1, 0.6, 0.0]],
                                    [[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.3, 0.2, 0.1, 0.3, 0.1],
                                     [0.1, 0.05, 0.5, 0.3, 0.05],
                                     [0.1, 0.2, 0.1, 0.6, 0.0]]])
        targets = torch.Tensor([[0, 3, 2, 1, 3],
                                [3, 1, 3, 3, 3]])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 5 / 6)

        # Test the same thing but with a mask:
        mask = torch.Tensor([[0, 1, 1, 1, 0],
                             [1, 0, 1, 0, 0]])
        accuracy(predictions, targets, mask)
        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 2 / 3)

    def test_top_k_label_categorical_accuracy_catches_exceptions(self):
        accuracy = LabelCategoricalAccuracy(positive_label=3)
        predictions = torch.rand([5, 7])
        out_of_range_labels = torch.Tensor([10, 3, 4, 0, 1])
        with pytest.raises(ConfigurationError):
            accuracy(predictions, out_of_range_labels)
