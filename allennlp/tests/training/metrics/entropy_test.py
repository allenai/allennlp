import torch
from torch.testing import assert_allclose

from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.training.metrics import Entropy


class EntropyTest(AllenNlpTestCase):
    @multi_device
    def test_low_entropy_distribution(self, device: str):
        metric = Entropy()
        logits = torch.tensor(
            [[10000, -10000, -10000, -1000], [10000, -10000, -10000, -1000]],
            dtype=torch.float,
            device=device,
        )
        metric(logits)
        assert metric.get_metric() == 0.0

    @multi_device
    def test_entropy_for_uniform_distribution(self, device: str):
        metric = Entropy()
        logits = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float, device=device)
        metric(logits)
        assert_allclose(metric.get_metric(), torch.tensor(1.38629436, device=device))
        # actual values shouldn't effect uniform distribution:
        logits = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 2]], dtype=torch.float, device=device)
        metric(logits)
        assert_allclose(metric.get_metric(), torch.tensor(1.38629436, device=device))

        metric.reset()
        assert metric._entropy == 0.0
        assert metric._count == 0.0

    @multi_device
    def test_masked_case(self, device: str):
        metric = Entropy()
        # This would have non-zero entropy without the mask.
        logits = torch.tensor(
            [[1, 1, 1, 1], [10000, -10000, -10000, -1000]], dtype=torch.float, device=device
        )
        mask = torch.tensor([False, True], device=device)
        metric(logits, mask)
        assert metric.get_metric() == 0.0
