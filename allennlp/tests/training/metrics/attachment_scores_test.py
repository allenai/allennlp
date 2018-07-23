# pylint: disable=no-self-use,invalid-name,protected-access
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import AttachmentScores


class AttachmentScoresTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.scorer = AttachmentScores()

        self.predictions = torch.Tensor([[0, 1, 3, 5, 2, 4],
                                         [0, 3, 2, 1, 0, 0]])

        self.gold_indices = torch.Tensor([[0, 1, 3, 5, 2, 4],
                                          [0, 3, 2, 1, 0, 0]])

        self.label_predictions = torch.Tensor([[0, 5, 2, 1, 4, 2],
                                               [0, 4, 8, 2, 0, 0]])

        self.gold_labels = torch.Tensor([[0, 5, 2, 1, 4, 2],
                                         [0, 4, 8, 2, 0, 0]])

        self.mask = torch.Tensor([[1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 0, 0]])

    def test_perfect_scores(self):
        self.scorer(self.predictions, self.label_predictions,
                    self.gold_indices, self.gold_labels, self.mask)

        for value in self.scorer.get_metric().values():
            assert value == 1.0

    def test_unlabeled_accuracy_ignores_incorrect_labels(self):
        label_predictions = self.label_predictions
        # Change some stuff so our 4 of our label predictions are wrong.
        label_predictions[0, 3:] = 3
        label_predictions[1, 0] = 7
        self.scorer(self.predictions, label_predictions,
                    self.gold_indices, self.gold_labels, self.mask)

        metrics = self.scorer.get_metric()

        assert metrics["UAS"] == 1.0
        assert metrics["UEM"] == 1.0

        # 4 / 12 labels were wrong and 2 positions
        # are masked, so 6/10 = 0.6 LAS.
        assert metrics["LAS"] == 0.6
        # Neither should have labeled exact match.
        assert metrics["LEM"] == 0.0

    def test_labeled_accuracy_is_affected_by_incorrect_heads(self):
        predictions = self.predictions
        # Change some stuff so our 4 of our predictions are wrong.
        predictions[0, 3:] = 3
        predictions[1, 0] = 7
        # This one is in the padded part, so it shouldn't affect anything.
        predictions[1, 5] = 7
        self.scorer(predictions, self.label_predictions,
                    self.gold_indices, self.gold_labels, self.mask)

        metrics = self.scorer.get_metric()

        # 4 heads are incorrect, so the unlabeled score should be
        # 6/10 = 0.6 LAS.
        assert metrics["UAS"] == 0.6
        # All the labels were correct, but some heads
        # were wrong, so the LAS should equal the UAS.
        assert metrics["LAS"] == 0.6

        # Neither batch element had a perfect labeled or unlabeled EM.
        assert metrics["LEM"] == 0.0
        assert metrics["UEM"] == 0.0
