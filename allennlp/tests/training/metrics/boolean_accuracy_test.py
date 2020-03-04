import torch
import pytest

from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.training.metrics import BooleanAccuracy


class BooleanAccuracyTest(AllenNlpTestCase):
    @multi_device
    def test_accuracy_computation(self, device: str):
        accuracy = BooleanAccuracy()
        predictions = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], device=device)
        targets = torch.tensor([[0, 1], [2, 2], [4, 5], [7, 7]], device=device)
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 2 / 4

        mask = torch.ones(4, 2, device=device).bool()
        mask[1, 1] = 0
        accuracy(predictions, targets, mask)
        assert accuracy.get_metric() == 5 / 8

        targets[1, 1] = 3
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 8 / 12

        accuracy.reset()
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 3 / 4

    @multi_device
    def test_skips_completely_masked_instances(self, device: str):
        accuracy = BooleanAccuracy()
        predictions = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], device=device)
        targets = torch.tensor([[0, 1], [2, 2], [4, 5], [7, 7]], device=device)

        mask = torch.tensor(
            [[False, False], [True, False], [True, True], [True, True]], device=device
        )
        accuracy(predictions, targets, mask)

        # First example should be skipped, second is correct with mask, third is correct, fourth is wrong.
        assert accuracy.get_metric() == 2 / 3

    @multi_device
    def test_incorrect_gold_labels_shape_catches_exceptions(self, device: str):
        accuracy = BooleanAccuracy()
        predictions = torch.rand([5, 7], device=device)
        incorrect_shape_labels = torch.rand([5, 8], device=device)
        with pytest.raises(ValueError):
            accuracy(predictions, incorrect_shape_labels)

    @multi_device
    def test_incorrect_mask_shape_catches_exceptions(self, device: str):
        accuracy = BooleanAccuracy()
        predictions = torch.rand([5, 7], device=device)
        labels = torch.rand([5, 7], device=device)
        incorrect_shape_mask = torch.randint(0, 2, [5, 8], device=device).bool()
        with pytest.raises(ValueError):
            accuracy(predictions, labels, incorrect_shape_mask)

    @multi_device
    def test_does_not_divide_by_zero_with_no_count(self, device: str):
        accuracy = BooleanAccuracy()
        self.assertAlmostEqual(accuracy.get_metric(), 0.0)
