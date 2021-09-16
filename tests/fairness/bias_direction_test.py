import torch
from torch import allclose
import pytest
import math

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.fairness.bias_direction import (
    PCABiasDirection,
    PairedPCABiasDirection,
    TwoMeansBiasDirection,
    ClassificationNormalBiasDirection,
)


class PCABiasDirectionTest(AllenNlpTestCase):
    def test_pca_invalid_dims(self):
        pca = PCABiasDirection()
        with pytest.raises(ConfigurationError):
            pca(torch.zeros(2))

    @multi_device
    def test_pca_without_grad(self, device: str):
        seed_embeddings = torch.eye(2, device=device)
        pca = PCABiasDirection()

        const = 1 / math.sqrt(2)
        expected_bias_direction = torch.tensor([const, -const], device=device)
        test_bias_direction = pca(seed_embeddings)
        k = expected_bias_direction / test_bias_direction
        assert k[0].item() == pytest.approx(k[1].item())
        assert seed_embeddings.grad is None

    @multi_device
    def test_pca_with_grad(self, device: str):
        # add noise to avoid "RuntimeError: triangular_solve_cpu: U(2,2) is zero, singular U."
        torch.manual_seed(0)
        seed_embeddings = torch.eye(2, device=device) + (1 - torch.eye(2, device=device)) * 1e-1
        seed_embeddings = seed_embeddings.requires_grad_()
        assert seed_embeddings.grad is None

        pca = PCABiasDirection(requires_grad=True)
        test_bias_direction = pca(seed_embeddings)
        test_bias_direction.sum().backward()
        assert seed_embeddings.grad is not None


class PairedPCABiasDirectionTest(AllenNlpTestCase):
    def test_paired_pca_invalid_dims(self):
        paired_pca = PairedPCABiasDirection()
        with pytest.raises(ConfigurationError):
            paired_pca(torch.zeros(2), torch.zeros(3))

        with pytest.raises(ConfigurationError):
            paired_pca(torch.zeros(2), torch.zeros(2))

    @multi_device
    def test_paired_pca_without_grad(self, device: str):
        seed_embeddings1 = torch.tensor([[1.0, 0.5], [1.5, 1.0]], device=device)
        seed_embeddings2 = torch.tensor([[0.5, 1.0], [1.0, 1.5]], device=device)
        paired_pca = PairedPCABiasDirection()

        const = math.sqrt(2) / 2
        expected_bias_direction = torch.tensor([-const, const], device=device)
        test_bias_direction = paired_pca(seed_embeddings1, seed_embeddings2)
        k = expected_bias_direction / test_bias_direction
        assert k[0].item() == pytest.approx(k[1].item())
        assert seed_embeddings1.grad is None
        assert seed_embeddings2.grad is None

    @multi_device
    def test_paired_pca_with_grad(self, device: str):
        # add noise to avoid "RuntimeError: triangular_solve_cpu: U(2,2) is zero, singular U."
        torch.manual_seed(0)
        seed_embeddings1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device)
        seed_embeddings2 = (1 - torch.eye(2, device=device)) * 9e-1
        seed_embeddings1 = seed_embeddings1.requires_grad_()
        seed_embeddings2 = seed_embeddings2.requires_grad_()
        assert seed_embeddings1.grad is None
        assert seed_embeddings2.grad is None

        paired_pca = PairedPCABiasDirection(requires_grad=True)
        test_bias_direction = paired_pca(seed_embeddings1, seed_embeddings2)
        test_bias_direction.sum().backward()
        assert seed_embeddings1.grad is not None
        assert seed_embeddings2.grad is not None


class TwoMeansBiasDirectionTest(AllenNlpTestCase):
    def test_two_means_invalid_dims(self):
        two_means = TwoMeansBiasDirection()
        with pytest.raises(ConfigurationError):
            two_means(torch.zeros(2), torch.zeros(2))

        with pytest.raises(ConfigurationError):
            two_means(torch.zeros(2, 2), torch.zeros(2, 3))

    @multi_device
    def test_two_means_without_grad(self, device: str):
        seed_embeddings1 = torch.eye(2, device=device)
        seed_embeddings2 = 1 - torch.eye(2, device=device)
        two_means = TwoMeansBiasDirection()

        expected_bias_direction = torch.tensor([float("nan"), float("nan")], device=device)
        test_bias_direction = two_means(seed_embeddings1, seed_embeddings2)
        assert allclose(expected_bias_direction, test_bias_direction, equal_nan=True)
        assert seed_embeddings1.grad is None
        assert seed_embeddings2.grad is None

    @multi_device
    def test_two_means_with_grad(self, device: str):
        seed_embeddings1 = torch.eye(2, device=device)
        seed_embeddings2 = 1 - torch.eye(2, device=device)
        seed_embeddings1 = seed_embeddings1.requires_grad_()
        seed_embeddings2 = seed_embeddings2.requires_grad_()
        assert seed_embeddings1.grad is None
        assert seed_embeddings2.grad is None

        two_means = TwoMeansBiasDirection(requires_grad=True)
        test_bias_direction = two_means(seed_embeddings1, seed_embeddings2)
        test_bias_direction.sum().backward()
        assert seed_embeddings1.grad is not None
        assert seed_embeddings2.grad is not None


class ClassificationNormalBiasDirectionTest(AllenNlpTestCase):
    def test_classification_normal_invalid_dims(self):
        classification_normal = ClassificationNormalBiasDirection()
        with pytest.raises(ConfigurationError):
            classification_normal(torch.zeros(2), torch.zeros(2))

        with pytest.raises(ConfigurationError):
            classification_normal(torch.zeros(2, 2), torch.zeros(2, 3))

    @multi_device
    def test_classification_normal_without_grad(self, device: str):
        seed_embeddings1 = torch.eye(2, device=device)
        seed_embeddings2 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device)
        classification_normal = ClassificationNormalBiasDirection()
        test_bias_direction = classification_normal(seed_embeddings1, seed_embeddings2)
        const = 1 / math.sqrt(2)
        assert (
            allclose(test_bias_direction, torch.Tensor([const, const]).to(device))
            or allclose(test_bias_direction, torch.Tensor([-const, -const]).to(device))
            or allclose(test_bias_direction, torch.Tensor([const, -const]).to(device))
            or allclose(test_bias_direction, torch.Tensor([-const, const]).to(device))
        )
        assert seed_embeddings1.grad is None
        assert seed_embeddings2.grad is None
