import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.sanity_checks.batch_norm_verification import BatchNormVerification


class BiasBatchNormModel(torch.nn.Module):
    def __init__(self, use_bias=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, kernel_size=1, bias=use_bias)
        self.bn = torch.nn.BatchNorm2d(5)

    def forward(self, x):
        # x: (B, 3, H, W)
        return self.bn(self.conv(x))


class TestBatchNormVerification(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.model_with_bias = BiasBatchNormModel(use_bias=True)
        self.model_without_bias = BiasBatchNormModel(use_bias=False)

        self.verification_with_bias = BatchNormVerification(self.model_with_bias)
        self.verification_without_bias = BatchNormVerification(self.model_without_bias)

        inputs = torch.rand(2, 3, 1, 4)

        self.valid_with_bias = self.verification_with_bias.check(inputs=inputs)
        self.valid_without_bias = self.verification_without_bias.check(inputs=inputs)

    def test_verification_check(self):
        assert not self.valid_with_bias
        assert self.valid_without_bias

    def test_collect_detections(self):
        detected_pairs = self.verification_with_bias.collect_detections()
        assert len(detected_pairs) == 1
        assert detected_pairs[0] == ("conv", "bn")

    def test_destroy_hooks(self):
        self.verification_with_bias.register_hooks()
        assert len(self.verification_with_bias._hook_handles) == 3

        for name, module in self.verification_with_bias.model.named_modules():
            assert module._forward_hooks

        self.verification_with_bias.destroy_hooks()
        assert len(self.verification_with_bias._hook_handles) == 0

        for name, module in self.verification_with_bias.model.named_modules():
            assert not module._forward_hooks
