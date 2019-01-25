# pylint: disable=no-self-use,invalid-name,protected-access
import torch

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.sampled_softmax_loss import _choice, SampledSoftmaxLoss
from allennlp.models.language_model import _SoftmaxLoss


class TestSampledSoftmaxLoss(AllenNlpTestCase):
    def test_choice(self):
        sample, num_tries = _choice(num_words=1000, num_samples=50)
        assert len(set(sample)) == 50
        assert all(0 <= x < 1000 for x in sample)
        assert num_tries >= 50

    def test_sampled_softmax_can_run(self):
        softmax = SampledSoftmaxLoss(num_words=1000, embedding_dim=12, num_samples=50)

        # sequence_length, embedding_dim
        embedding = torch.rand(100, 12)
        targets = torch.randint(0, 1000, (100,)).long()

        _ = softmax(embedding, targets)

    @pytest.mark.skip("this test is broken, joelg is working on fixing it")
    def test_sampled_almost_equals_unsampled_when_num_samples_is_almost_all(self):
        sampled_softmax = SampledSoftmaxLoss(num_words=10000, embedding_dim=12, num_samples=9999)
        unsampled_softmax = _SoftmaxLoss(num_words=10000, embedding_dim=12)

        # sequence_length, embedding_dim
        embedding = torch.rand(100, 12)
        targets = torch.randint(0, 1000, (100,)).long()

        full_loss = unsampled_softmax(embedding, targets).item()
        sampled_loss = sampled_softmax(embedding, targets).item()

        # Should be really close
        pct_error = (sampled_loss - full_loss) / full_loss
        assert abs(pct_error) < 0.02
