import pytest
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.fairness.fairness_metrics import Independence, Separation, Sufficiency


class IndependenceTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.independence = Independence()

    def test_invalid_dimensions(self):
        C = torch.eye(3).long()
        A = torch.eye(4).long()
        with pytest.raises(ConfigurationError):
            self.independence(C, A)

    def test_invalid_num_classes(self):
        C = torch.eye(3).long()
        A = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            self.independence(C, A, 1)

    def test_independence_num_classes_inferred(self):
        A = torch.eye(3).long()
        C = 2 * A

        # P(C) = [0.667, 0.0, 0.333]
        # P(C | A = 0) = [1.0, 0.0, 0.0]
        # P(C | A = 1) = [0.0, 0.0, 1.0]
        # KL(P(C | A = 0) || P(C)) = 0.4055
        # KL(P(C | A = 1) || P(C)) = 1.0986
        expected_kl_divs = {0: 0.4055, 1: 1.0986}
        test_kl_divs = {k: v.item() for k, v in self.independence(C, A).items()}
        assert expected_kl_divs == pytest.approx(test_kl_divs, abs=1e-3)

    def test_independence_num_classes_supplied(self):
        A = torch.eye(3).long()
        C = 2 * A
        num_classes = 4

        # P(C) = [0.667, 0.0, 0.333, 0.0]
        # P(C | A = 0) = [1.0, 0.0, 0.0, 0.0]
        # P(C | A = 1) = [0.0, 0.0, 1.0, 0.0]
        # KL(P(C | A = 0) || P(C)) = 0.4055
        # KL(P(C | A = 1) || P(C)) = 1.0986
        expected_kl_divs = {0: 0.4055, 1: 1.0986}
        test_kl_divs = {k: v.item() for k, v in self.independence(C, A, num_classes).items()}
        assert expected_kl_divs == pytest.approx(test_kl_divs, abs=1e-3)


class SeparationTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.separation = Separation()

    def test_invalid_dimensions(self):
        C = torch.eye(3).long()
        Y = torch.eye(4).long()
        A = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            self.separation(C, Y, A)

    def test_invalid_num_classes(self):
        C = 2 * torch.eye(3).long()
        Y = torch.eye(3).long()
        A = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            self.separation(C, Y, A)

    def test_separation(self):
        C = torch.eye(3).long()
        Y = C
        A = C

        # P(C | Y = 0) = [1.0, 0.0]
        # P(C | A = 0, Y = 0) = [1.0, 0.0]
        # P(C | A = 1, Y = 0) = [0.0, 0.0]
        # KL(P(C | A = 0, Y = 0) || P(C | Y = 0)) = 0.0
        # KL(P(C | A = 1, Y = 0) || P(C | Y = 0)) = 0.0

        # P(C | Y = 1) = [0.0, 1.0]
        # P(C | A = 0, Y = 1) = [0.0, 0.0]
        # P(C | A = 1, Y = 1) = [0.0, 1.0]
        # KL(P(C | A = 0, Y = 1) || P(C | Y = 1)) = 0.0
        # KL(P(C | A = 1, Y = 1) || P(C | Y = 1)) = 0.0

        expected_kl_divs = {0: {0: 0.0, 1: 0.0}, 1: {0: 0.0, 1: 0.0}}
        test_kl_divs = {
            k1: {k2: v2.item() for k2, v2 in v1.items()}
            for k1, v1 in self.separation(C, Y, A).items()
        }
        assert expected_kl_divs == test_kl_divs


class SufficiencyTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.sufficiency = Sufficiency()

    def test_invalid_dimensions(self):
        C = torch.eye(3).long()
        Y = torch.eye(4).long()
        A = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            self.sufficiency(C, Y, A)

    def test_invalid_num_classes(self):
        C = 2 * torch.eye(3).long()
        Y = torch.eye(3).long()
        A = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            self.sufficiency(C, Y, A)

    def test_sufficiency(self):
        # Tests when C = 1 is not predicted
        C = torch.zeros(3, 3).long()
        Y = torch.eye(3).long()
        A = Y

        # P(Y | C = 0) = [0.667, 0.333]
        # P(Y | A = 0, C = 0) = [1.0, 0.0]
        # P(Y | A = 1, C = 0) = [0.0, 1.0]
        # KL(P(Y | A = 0, C = 0) || P(Y | C = 0)) = 0.4055
        # KL(P(Y | A = 1, C = 0) || P(Y | C = 0)) = 1.0986

        # P(Y | C = 1) = [0.0, 0.0]
        # P(Y | A = 0, C = 1) = [0.0, 0.0]
        # P(Y | A = 1, C = 1) = [0.0, 0.0]
        # KL(P(Y | A = 0, C = 1) || P(Y | C = 1)) = 0.0
        # KL(P(Y | A = 1, C = 1) || P(Y | C = 1)) = 0.0

        expected_kl_divs = {0: {0: 0.4055, 1: 1.0986}, 1: {0: 0.0, 1: 0.0}}
        test_kl_divs = {
            k1: {k2: v2.item() for k2, v2 in v1.items()}
            for k1, v1 in self.sufficiency(C, Y, A).items()
        }
        assert len(expected_kl_divs) == len(test_kl_divs)
        assert expected_kl_divs[0] == pytest.approx(test_kl_divs[0], abs=1e-3)
        assert expected_kl_divs[1] == pytest.approx(test_kl_divs[1], abs=1e-3)
