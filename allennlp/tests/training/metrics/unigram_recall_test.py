import torch
from torch.testing import assert_allclose

from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.training.metrics import UnigramRecall


class UnigramRecallTest(AllenNlpTestCase):
    @multi_device
    def test_sequence_recall(self, device: str):
        recall = UnigramRecall()
        gold = torch.tensor([[1, 2, 3], [2, 4, 8], [7, 1, 1]], device=device)
        predictions = torch.tensor(
            [[[1, 2, 3], [1, 2, -1]], [[2, 4, 8], [2, 5, 9]], [[-1, -1, -1], [7, 1, -1]]],
            device=device,
        )

        recall(predictions, gold)
        actual_recall = recall.get_metric()
        assert_allclose(actual_recall, 1)

    @multi_device
    def test_sequence_recall_respects_mask(self, device: str):
        recall = UnigramRecall()
        gold = torch.tensor([[2, 4, 8], [1, 2, 3], [7, 1, 1], [11, 14, 17]], device=device)
        predictions = torch.tensor(
            [
                [[2, 4, 8], [2, 5, 9]],  # 3/3
                [[-1, 2, 4], [3, 8, -1]],  # 2/2
                [[-1, -1, -1], [7, 2, -1]],  # 1/2
                [[12, 13, 17], [11, 13, 18]],  # 2/2
            ],
            device=device,
        )
        mask = torch.tensor(
            [[True, True, True], [False, True, True], [True, True, False], [True, False, True]],
            device=device,
        )

        recall(predictions, gold, mask)
        actual_recall = recall.get_metric()
        assert_allclose(actual_recall, 7 / 8)

    @multi_device
    def test_sequence_recall_accumulates_and_resets_correctly(self, device: str):
        recall = UnigramRecall()
        gold = torch.tensor([[1, 2, 3]], device=device)
        recall(torch.tensor([[[1, 2, 3]]], device=device), gold)
        recall(torch.tensor([[[7, 8, 4]]], device=device), gold)

        actual_recall = recall.get_metric(reset=True)
        assert_allclose(actual_recall, 1 / 2)
        assert recall.correct_count == 0
        assert recall.total_count == 0

    @multi_device
    def test_get_metric_on_new_object_works(self, device: str):
        recall = UnigramRecall()

        actual_recall = recall.get_metric(reset=True)
        assert_allclose(actual_recall, 0)
