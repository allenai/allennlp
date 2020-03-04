import pytest
import torch
from torch.testing import assert_allclose

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.training.metrics import CategoricalAccuracy


class CategoricalAccuracyTest(AllenNlpTestCase):
    @multi_device
    def test_categorical_accuracy(self, device: str):
        accuracy = CategoricalAccuracy()
        predictions = torch.tensor(
            [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0]], device=device
        )
        targets = torch.tensor([0, 3], device=device)
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 0.50

    @multi_device
    def test_top_k_categorical_accuracy(self, device: str):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.tensor(
            [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0]], device=device
        )
        targets = torch.tensor([0, 3], device=device)
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 1.0

    @multi_device
    def test_top_k_categorical_accuracy_accumulates_and_resets_correctly(self, device: str):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.tensor(
            [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0]], device=device
        )
        targets = torch.tensor([0, 3], device=device)
        accuracy(predictions, targets)
        accuracy(predictions, targets)
        accuracy(predictions, torch.tensor([4, 4], device=device))
        accuracy(predictions, torch.tensor([4, 4], device=device))
        actual_accuracy = accuracy.get_metric(reset=True)
        assert actual_accuracy == 0.50
        assert accuracy.correct_count == 0.0
        assert accuracy.total_count == 0.0

    @multi_device
    def test_top_k_categorical_accuracy_respects_mask(self, device: str):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.tensor(
            [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.2, 0.5, 0.2, 0.0]],
            device=device,
        )
        targets = torch.tensor([0, 3, 0], device=device)
        mask = torch.tensor([False, True, True], device=device)
        accuracy(predictions, targets, mask)
        actual_accuracy = accuracy.get_metric()
        assert_allclose(actual_accuracy, 0.50)

    @multi_device
    def test_top_k_categorical_accuracy_works_for_sequences(self, device: str):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.tensor(
            [
                [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]],
                [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]],
            ],
            device=device,
        )
        targets = torch.tensor([[0, 3, 4], [0, 1, 4]], device=device)
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric(reset=True)
        assert_allclose(actual_accuracy, 0.6666666)

        # Test the same thing but with a mask:
        mask = torch.tensor([[False, True, True], [True, False, True]], device=device)
        accuracy(predictions, targets, mask)
        actual_accuracy = accuracy.get_metric(reset=True)
        assert_allclose(actual_accuracy, 0.50)

    @multi_device
    def test_top_k_categorical_accuracy_catches_exceptions(self, device: str):
        accuracy = CategoricalAccuracy()
        predictions = torch.rand([5, 7], device=device)
        out_of_range_labels = torch.tensor([10, 3, 4, 0, 1], device=device)
        with pytest.raises(ConfigurationError):
            accuracy(predictions, out_of_range_labels)

    @multi_device
    def test_tie_break_categorical_accuracy(self, device: str):
        accuracy = CategoricalAccuracy(tie_break=True)
        predictions = torch.tensor(
            [[0.35, 0.25, 0.35, 0.35, 0.35], [0.1, 0.6, 0.1, 0.2, 0.2], [0.1, 0.0, 0.1, 0.2, 0.2]],
            device=device,
        )
        # Test without mask:
        targets = torch.tensor([2, 1, 4], device=device)
        accuracy(predictions, targets)
        assert accuracy.get_metric(reset=True) == (0.25 + 1 + 0.5) / 3.0

        # # # Test with mask
        mask = torch.tensor([True, False, True], device=device)
        targets = torch.tensor([2, 1, 4], device=device)
        accuracy(predictions, targets, mask)
        assert accuracy.get_metric(reset=True) == (0.25 + 0.5) / 2.0

        # # Test tie-break with sequence
        predictions = torch.tensor(
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
            ],
            device=device,
        )
        targets = torch.tensor(
            [[0, 1, 3], [0, 3, 4]], device=device  # 0.25 + 1 + 0.5  # 0.25 + 0 + 0.5 = 2.5
        )
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric(reset=True)
        assert_allclose(actual_accuracy, 2.5 / 6.0)

    @multi_device
    def test_top_k_and_tie_break_together_catches_exceptions(self, device: str):
        with pytest.raises(ConfigurationError):
            CategoricalAccuracy(top_k=2, tie_break=True)

    @multi_device
    def test_incorrect_top_k_catches_exceptions(self, device: str):
        with pytest.raises(ConfigurationError):
            CategoricalAccuracy(top_k=0)

    @multi_device
    def test_does_not_divide_by_zero_with_no_count(self, device: str):
        accuracy = CategoricalAccuracy()
        self.assertAlmostEqual(accuracy.get_metric(), 0.0)
