import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.testing.confidence_check_test import (
    FakeModelForTestingNormalizationBiasVerification,
)
from allennlp.confidence_checks.normalization_bias_verification import NormalizationBiasVerification


class TestNormalizationBiasVerification(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.model_with_bias = FakeModelForTestingNormalizationBiasVerification(use_bias=True)
        self.model_without_bias = FakeModelForTestingNormalizationBiasVerification(use_bias=False)

        self.verification_with_bias = NormalizationBiasVerification(self.model_with_bias)
        self.verification_without_bias = NormalizationBiasVerification(self.model_without_bias)

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
