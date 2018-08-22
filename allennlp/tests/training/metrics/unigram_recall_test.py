# pylint: disable=no-self-use,invalid-name,protected-access
import torch
import pytest
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import UnigramRecall

class UnigramRecallTest(AllenNlpTestCase):
    def test_sequence_recall(self):
        recall = UnigramRecall()
        gold = torch.tensor([
            [1, 2, 3],
            [2, 4, 8],
            [0, 1, 1]
        ])
        predictions = torch.tensor([
            [[1, 2, 3], [1, 2, -1]],
            [[2, 4, 8], [2, 5, 9]],
            [[-1, -1, -1], [0, 1, -1]]
        ])

        recall(predictions, gold)
        actual_recall = recall.get_metric()
        numpy.testing.assert_almost_equal(actual_recall, 2/3)

    def test_sequence_recall_respects_mask(self):
        recall = UnigramRecall()
        gold = torch.tensor([
            [1, 2, 3],
            [2, 4, 8],
            [0, 1, 1],
            [11, 13, 17],
        ])
        predictions = torch.tensor([
            [[1, 2, 3], [1, 2, -1]],
            [[2, 4, 8], [2, 5, 9]],
            [[-1, -1, -1], [0, 1, -1]],
            [[12, 13, 17], [11, 13, 18]]
        ])
        mask = torch.tensor([
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1]
        ])

        recall(predictions, gold, mask)
        actual_recall = recall.get_metric()
        numpy.testing.assert_almost_equal(actual_recall, 3/4)

    def test_sequence_recall_accumulates_and_resets_correctly(self):
        recall = UnigramRecall()
        gold = torch.tensor([
            [1, 2, 3],
        ])
        recall(torch.tensor([[[1, 2, 3]]]), gold)
        recall(torch.tensor([[[1, 2, 4]]]), gold)

        actual_recall = recall.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_recall, 1/2)
        assert recall.correct_count == 0
        assert recall.total_count == 0
