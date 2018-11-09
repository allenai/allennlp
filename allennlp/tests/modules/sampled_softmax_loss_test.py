# pylint: disable=no-self-use,invalid-name,protected-access
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.sampled_softmax_loss import _choice, SampledSoftmaxLoss


class TestSampledSoftmax(AllenNlpTestCase):
    def test_choice(self):
        sample, num_samples = _choice(num_words=1000, num_samples=50)
        assert len(set(sample)) == 50
        assert all(0 <= x < 1000 for x in sample)
        assert num_samples >= 50

    def test_sampled_softmax(self):
        softmax = SampledSoftmaxLoss(num_words=1000, embedding_dim=12, num_samples=50)

        # sequence_length, embedding_dim
        embedding = torch.rand(100, 12)
        targets = torch.randint(0, 1000, (100,)).long()

        _ = softmax(embedding, targets)
