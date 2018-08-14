# pylint: disable=no-self-use,invalid-name,protected-access
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import ConllCorefScores

class ConllCorefScoresTest(AllenNlpTestCase):
    def test_get_predicted_clusters(self):
        top_spans = torch.Tensor([[0, 1], [4, 6], [8, 9]]).long()
        antecedent_indices = torch.Tensor([[-1, -1, -1],
                                           [0, -1, -1],
                                           [0, 1, -1]]).long()
        predicted_antecedents = torch.Tensor([-1, -1, 1]).long()
        clusters, mention_to_cluster = ConllCorefScores.get_predicted_clusters(top_spans,
                                                                               antecedent_indices,
                                                                               predicted_antecedents)
        assert len(clusters) == 1
        assert set(clusters[0]) == {(4, 6), (8, 9)}
        assert mention_to_cluster == {(4, 6): clusters[0], (8, 9): clusters[0]}
