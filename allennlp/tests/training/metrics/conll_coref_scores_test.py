import torch

from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.training.metrics import ConllCorefScores


class ConllCorefScoresTest(AllenNlpTestCase):
    @multi_device
    def test_get_predicted_clusters(self, device: str):
        top_spans = torch.tensor([[0, 1], [4, 6], [8, 9]], device=device)
        antecedent_indices = torch.tensor([[-1, -1, -1], [0, -1, -1], [0, 1, -1]], device=device)
        predicted_antecedents = torch.tensor([-1, -1, 1], device=device)
        clusters, mention_to_cluster = ConllCorefScores.get_predicted_clusters(
            top_spans, antecedent_indices, predicted_antecedents
        )
        assert len(clusters) == 1
        assert set(clusters[0]) == {(4, 6), (8, 9)}
        assert mention_to_cluster == {(4, 6): clusters[0], (8, 9): clusters[0]}
