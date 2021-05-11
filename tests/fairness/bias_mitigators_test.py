import torch
from torch import allclose
import pytest
import json

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.fairness.bias_mitigators import (
    LinearBiasMitigator,
    HardBiasMitigator,
    INLPBiasMitigator,
    OSCaRBiasMitigator,
)
from allennlp.fairness.bias_direction import TwoMeansBiasDirection


class LinearBiasMitigatorTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        # embedding data from VERB demo
        emb_filename = str(self.FIXTURES_ROOT / "fairness" / "bias_embeddings.json")
        with open(emb_filename) as emb_file:
            emb_data = json.load(emb_file)

        seed_embeddings1 = torch.cat(
            [
                torch.Tensor(emb_data["he"]).reshape(1, -1),
                torch.Tensor(emb_data["him"]).reshape(1, -1),
            ]
        )
        seed_embeddings2 = torch.cat(
            [
                torch.Tensor(emb_data["she"]).reshape(1, -1),
                torch.Tensor(emb_data["her"]).reshape(1, -1),
            ]
        )
        tm = TwoMeansBiasDirection()
        self.bias_direction = tm(seed_embeddings1, seed_embeddings2)

        evaluation_embeddings = []
        expected_bias_mitigated_embeddings = []
        for word in ["engineer", "banker", "nurse", "receptionist"]:
            evaluation_embeddings.append(torch.Tensor(emb_data[word]).reshape(1, -1))
            expected_bias_mitigated_embeddings.append(
                torch.Tensor(emb_data["linear_two_means_" + word]).reshape(1, -1)
            )
        self.evaluation_embeddings = torch.cat(evaluation_embeddings).reshape(2, 2, -1)
        self.expected_bias_mitigated_embeddings = torch.cat(
            expected_bias_mitigated_embeddings
        ).reshape(2, 2, -1)

    def test_invalid_dims(self):
        lbm = LinearBiasMitigator()
        with pytest.raises(ConfigurationError):
            lbm(torch.zeros(2), torch.zeros(2))
        with pytest.raises(ConfigurationError):
            lbm(torch.zeros(2), torch.zeros((2, 2)))
        with pytest.raises(ConfigurationError):
            lbm(torch.zeros((2, 3)), torch.zeros(2))

    @multi_device
    def test_lbm_without_grad(self, device: str):
        self.bias_direction = self.bias_direction.to(device)
        self.evaluation_embeddings = self.evaluation_embeddings.to(device)
        self.expected_bias_mitigated_embeddings = self.expected_bias_mitigated_embeddings.to(device)

        lbm = LinearBiasMitigator()
        test_bias_mitigated_embeddings = lbm(self.evaluation_embeddings, self.bias_direction)
        assert allclose(
            self.expected_bias_mitigated_embeddings, test_bias_mitigated_embeddings, atol=1e-6
        )

    @multi_device
    def test_lbm_with_grad(self, device: str):
        self.bias_direction = self.bias_direction.to(device).requires_grad_()
        self.evaluation_embeddings = self.evaluation_embeddings.to(device).requires_grad_()
        assert self.bias_direction.grad is None
        assert self.evaluation_embeddings.grad is None

        lbm = LinearBiasMitigator(requires_grad=True)
        test_bias_mitigated_embeddings = lbm(self.evaluation_embeddings, self.bias_direction)
        test_bias_mitigated_embeddings.sum().backward()
        assert self.bias_direction.grad is not None
        assert self.evaluation_embeddings.grad is not None


class HardBiasMitigatorTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        # embedding data from VERB demo
        emb_filename = str(self.FIXTURES_ROOT / "fairness" / "bias_embeddings.json")
        with open(emb_filename) as emb_file:
            emb_data = json.load(emb_file)

        seed_embeddings1 = torch.cat(
            [
                torch.Tensor(emb_data["he"]).reshape(1, -1),
                torch.Tensor(emb_data["man"]).reshape(1, -1),
            ]
        )
        seed_embeddings2 = torch.cat(
            [
                torch.Tensor(emb_data["she"]).reshape(1, -1),
                torch.Tensor(emb_data["woman"]).reshape(1, -1),
            ]
        )
        tm = TwoMeansBiasDirection()
        self.bias_direction = tm(seed_embeddings1, seed_embeddings2)

        self.equalize_embeddings1 = torch.cat(
            [
                torch.Tensor(emb_data["boy"]).reshape(1, -1),
                torch.Tensor(emb_data["brother"]).reshape(1, -1),
            ]
        ).unsqueeze(0)
        self.equalize_embeddings2 = torch.cat(
            [
                torch.Tensor(emb_data["girl"]).reshape(1, -1),
                torch.Tensor(emb_data["sister"]).reshape(1, -1),
            ]
        ).unsqueeze(0)

        evaluation_embeddings = []
        expected_bias_mitigated_embeddings = []
        for word in ["engineer", "banker", "nurse", "receptionist"]:
            evaluation_embeddings.append(torch.Tensor(emb_data[word]).reshape(1, -1))
            expected_bias_mitigated_embeddings.append(
                torch.Tensor(emb_data["hard_two_means_" + word]).reshape(1, -1)
            )
        for word in ["boy", "brother", "girl", "sister"]:
            expected_bias_mitigated_embeddings.append(
                torch.Tensor(emb_data["hard_two_means_" + word]).reshape(1, -1)
            )
        self.evaluation_embeddings = torch.cat(evaluation_embeddings).reshape(2, 2, -1)
        self.expected_bias_mitigated_embeddings = torch.cat(
            expected_bias_mitigated_embeddings
        ).reshape(4, 2, -1)

    def test_invalid_dims(self):
        hbm = HardBiasMitigator()
        with pytest.raises(ConfigurationError):
            hbm(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2))
        with pytest.raises(ConfigurationError):
            hbm(torch.zeros(2), torch.zeros(2), torch.zeros((2, 2)), torch.zeros((3, 2)))
        with pytest.raises(ConfigurationError):
            hbm(torch.zeros(2), torch.zeros(2), torch.zeros((2, 2)), torch.zeros((2, 2)))
        with pytest.raises(ConfigurationError):
            hbm(torch.zeros((3, 3)), torch.zeros(2), torch.zeros((2, 2)), torch.zeros((2, 2)))
        with pytest.raises(ConfigurationError):
            hbm(torch.zeros((3, 2)), torch.zeros((2, 2)), torch.zeros((2, 2)), torch.zeros((2, 2)))
        with pytest.raises(ConfigurationError):
            hbm(torch.zeros((3, 2)), torch.zeros(3), torch.zeros((2, 2)), torch.zeros((2, 2)))

    @multi_device
    def test_hbm_without_grad(self, device: str):
        self.bias_direction = self.bias_direction.to(device)
        self.evaluation_embeddings = self.evaluation_embeddings.to(device)
        self.equalize_embeddings1 = self.equalize_embeddings1.to(device)
        self.equalize_embeddings2 = self.equalize_embeddings2.to(device)
        self.expected_bias_mitigated_embeddings = self.expected_bias_mitigated_embeddings.to(device)

        hbm = HardBiasMitigator()
        test_bias_mitigated_embeddings = hbm(
            self.evaluation_embeddings,
            self.bias_direction,
            self.equalize_embeddings1,
            self.equalize_embeddings2,
        )
        assert allclose(
            self.expected_bias_mitigated_embeddings, test_bias_mitigated_embeddings, atol=1e-6
        )

    @multi_device
    def test_hbm_with_grad(self, device: str):
        self.bias_direction = self.bias_direction.to(device).requires_grad_()
        self.evaluation_embeddings = self.evaluation_embeddings.to(device).requires_grad_()
        self.equalize_embeddings1 = self.equalize_embeddings1.to(device).requires_grad_()
        self.equalize_embeddings2 = self.equalize_embeddings2.to(device).requires_grad_()
        assert self.bias_direction.grad is None
        assert self.evaluation_embeddings.grad is None
        assert self.equalize_embeddings1.grad is None
        assert self.equalize_embeddings2.grad is None

        hbm = HardBiasMitigator(requires_grad=True)
        test_bias_mitigated_embeddings = hbm(
            self.evaluation_embeddings,
            self.bias_direction,
            self.equalize_embeddings1,
            self.equalize_embeddings2,
        )
        test_bias_mitigated_embeddings.sum().backward()
        assert self.bias_direction.grad is not None
        assert self.evaluation_embeddings.grad is not None
        assert self.equalize_embeddings1.grad is not None
        assert self.equalize_embeddings2.grad is not None


class INLPBiasMitigatorTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        # embedding data from VERB demo
        emb_filename = str(self.FIXTURES_ROOT / "fairness" / "bias_embeddings.json")
        with open(emb_filename) as emb_file:
            emb_data = json.load(emb_file)

        seed_embeddings1 = []
        for word in ["man", "he", "his", "boy", "grandpa", "uncle", "jack"]:
            seed_embeddings1.append(torch.Tensor(emb_data[word]).reshape(1, -1))
        self.seed_embeddings1 = torch.cat(seed_embeddings1)

        seed_embeddings2 = []
        for word in ["woman", "she", "her", "girl", "grandma", "aunt", "jill"]:
            seed_embeddings2.append(torch.Tensor(emb_data[word]).reshape(1, -1))
        self.seed_embeddings2 = torch.cat(seed_embeddings2)

        evaluation_embeddings = []
        expected_bias_mitigated_embeddings = []
        for word in ["engineer", "homemaker"]:
            evaluation_embeddings.append(torch.Tensor(emb_data[word]).reshape(1, -1))
            expected_bias_mitigated_embeddings.append(
                torch.Tensor(emb_data["inlp_" + word]).reshape(1, -1)
            )
        self.evaluation_embeddings = torch.cat(evaluation_embeddings)
        self.expected_bias_mitigated_embeddings = torch.cat(expected_bias_mitigated_embeddings)

    def test_invalid_dims(self):
        ibm = INLPBiasMitigator()
        with pytest.raises(ConfigurationError):
            ibm(torch.zeros(2), torch.zeros(2), torch.zeros(2))
        with pytest.raises(ConfigurationError):
            ibm(torch.zeros(2), torch.zeros((2, 2)), torch.zeros((2, 3)))
        with pytest.raises(ConfigurationError):
            ibm(torch.zeros(2), torch.zeros((2, 2)), torch.zeros((2, 2)))
        with pytest.raises(ConfigurationError):
            ibm(torch.zeros((2, 3)), torch.zeros((2, 2)), torch.zeros((2, 2)))

    @multi_device
    def test_inlp(self, device: str):
        self.seed_embeddings1 = self.seed_embeddings1.to(device)
        self.seed_embeddings2 = self.seed_embeddings2.to(device)
        self.evaluation_embeddings = self.evaluation_embeddings.to(device)
        self.expected_bias_mitigated_embeddings = self.expected_bias_mitigated_embeddings.to(device)

        ibm = INLPBiasMitigator()
        test_bias_mitigated_embeddings = ibm(
            self.evaluation_embeddings, self.seed_embeddings1, self.seed_embeddings2
        )
        assert allclose(
            self.expected_bias_mitigated_embeddings, test_bias_mitigated_embeddings, atol=1e-6
        )


class OSCaRBiasMitigatorTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        # embedding data from VERB demo
        emb_filename = str(self.FIXTURES_ROOT / "fairness" / "bias_embeddings.json")
        with open(emb_filename) as emb_file:
            emb_data = json.load(emb_file)

        self.bias_direction1 = torch.Tensor(emb_data["oscar_bias1"])
        self.bias_direction2 = torch.Tensor(emb_data["oscar_bias2"])

        evaluation_embeddings = []
        expected_bias_mitigated_embeddings = []
        for word in ["programmer", "grandpa", "grandma"]:
            evaluation_embeddings.append(torch.Tensor(emb_data[word]).reshape(1, -1))
            expected_bias_mitigated_embeddings.append(
                torch.Tensor(emb_data["oscar_" + word]).reshape(1, -1)
            )
        self.evaluation_embeddings = torch.cat(evaluation_embeddings)
        self.expected_bias_mitigated_embeddings = torch.cat(expected_bias_mitigated_embeddings)

    def test_invalid_dims(self):
        ibm = INLPBiasMitigator()
        with pytest.raises(ConfigurationError):
            ibm(torch.zeros(2), torch.zeros(2), torch.zeros(2))
        with pytest.raises(ConfigurationError):
            ibm(torch.zeros(2), torch.zeros((2, 2)), torch.zeros((2, 3)))
        with pytest.raises(ConfigurationError):
            ibm(torch.zeros((2, 3)), torch.zeros(2), torch.zeros(2))
        with pytest.raises(ConfigurationError):
            ibm(torch.zeros((2, 1)), torch.zeros(1), torch.zeros(1))

    @multi_device
    def test_oscar_without_grad(self, device: str):
        self.bias_direction1 = self.bias_direction1.to(device)
        self.bias_direction2 = self.bias_direction2.to(device)
        self.evaluation_embeddings = self.evaluation_embeddings.to(device)
        self.expected_bias_mitigated_embeddings = self.expected_bias_mitigated_embeddings.to(device)

        obm = OSCaRBiasMitigator()
        test_bias_mitigated_embeddings = obm(
            self.evaluation_embeddings, self.bias_direction1, self.bias_direction2
        )
        assert allclose(
            self.expected_bias_mitigated_embeddings, test_bias_mitigated_embeddings, atol=1e-6
        )

    @multi_device
    def test_oscar_with_grad(self, device: str):
        self.bias_direction1 = self.bias_direction1.to(device).requires_grad_()
        self.bias_direction2 = self.bias_direction2.to(device).requires_grad_()
        self.evaluation_embeddings = self.evaluation_embeddings.to(device).requires_grad_()
        assert self.bias_direction1.grad is None
        assert self.bias_direction2.grad is None
        assert self.evaluation_embeddings.grad is None

        obm = OSCaRBiasMitigator(requires_grad=True)
        test_bias_mitigated_embeddings = obm(
            self.evaluation_embeddings, self.bias_direction1, self.bias_direction2
        )
        test_bias_mitigated_embeddings.sum().backward()
        assert self.bias_direction1.grad is not None
        assert self.bias_direction2.grad is not None
        assert self.evaluation_embeddings.grad is not None
