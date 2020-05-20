from typing import Tuple

import torch
import numpy as np

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.sampled_softmax_loss import _choice, SampledSoftmaxLoss
from allennlp.modules import SoftmaxLoss


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

    def test_sampled_equals_unsampled_during_eval(self):
        sampled_softmax = SampledSoftmaxLoss(num_words=10000, embedding_dim=12, num_samples=40)
        unsampled_softmax = SoftmaxLoss(num_words=10000, embedding_dim=12)

        sampled_softmax.eval()
        unsampled_softmax.eval()

        # set weights equal, use transpose because opposite shapes
        sampled_softmax.softmax_w.data = unsampled_softmax.softmax_w.t()
        sampled_softmax.softmax_b.data = unsampled_softmax.softmax_b

        # sequence_length, embedding_dim
        embedding = torch.rand(100, 12)
        targets = torch.randint(0, 1000, (100,)).long()

        full_loss = unsampled_softmax(embedding, targets).item()
        sampled_loss = sampled_softmax(embedding, targets).item()

        # Should be really close
        np.testing.assert_almost_equal(sampled_loss, full_loss)

    def test_sampled_softmax_has_greater_loss_in_train_mode(self):
        sampled_softmax = SampledSoftmaxLoss(num_words=10000, embedding_dim=12, num_samples=10)

        # sequence_length, embedding_dim
        embedding = torch.rand(100, 12)
        targets = torch.randint(0, 1000, (100,)).long()

        sampled_softmax.train()
        train_loss = sampled_softmax(embedding, targets).item()

        sampled_softmax.eval()
        eval_loss = sampled_softmax(embedding, targets).item()

        assert eval_loss > train_loss

    def test_sampled_equals_unsampled_when_biased_against_non_sampled_positions(self):
        sampled_softmax = SampledSoftmaxLoss(num_words=10000, embedding_dim=12, num_samples=10)
        unsampled_softmax = SoftmaxLoss(num_words=10000, embedding_dim=12)

        # fake out choice function
        FAKE_SAMPLES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 9999]

        def fake_choice(num_words: int, num_samples: int) -> Tuple[np.ndarray, int]:
            assert (num_words, num_samples) == (10000, 10)
            return np.array(FAKE_SAMPLES), 12

        sampled_softmax.choice_func = fake_choice

        # bias out the unsampled terms:
        for i in range(10000):
            if i not in FAKE_SAMPLES:
                unsampled_softmax.softmax_b[i] = -10000

        # set weights equal, use transpose because opposite shapes
        sampled_softmax.softmax_w.data = unsampled_softmax.softmax_w.t()
        sampled_softmax.softmax_b.data = unsampled_softmax.softmax_b

        sampled_softmax.train()
        unsampled_softmax.train()

        # sequence_length, embedding_dim
        embedding = torch.rand(100, 12)
        targets = torch.randint(0, 1000, (100,)).long()

        full_loss = unsampled_softmax(embedding, targets).item()
        sampled_loss = sampled_softmax(embedding, targets).item()

        # Should be close

        pct_error = (sampled_loss - full_loss) / full_loss
        assert abs(pct_error) < 0.001
