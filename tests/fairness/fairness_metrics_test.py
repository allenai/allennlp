import pytest
import torch
import math
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.fairness.fairness_metrics import (
    Independence,
    Separation,
    Sufficiency,
    DemographicParityWithoutGroundTruth,
)


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
        # KL(P(C | A = 1, Y = 0) || P(C | Y = 0)) = NaN

        # P(C | Y = 1) = [0.0, 1.0]
        # P(C | A = 0, Y = 1) = [0.0, 0.0]
        # P(C | A = 1, Y = 1) = [0.0, 1.0]
        # KL(P(C | A = 0, Y = 1) || P(C | Y = 1)) = NaN
        # KL(P(C | A = 1, Y = 1) || P(C | Y = 1)) = 0.0

        # KL divergence cannot be negative
        expected_kl_divs = {0: {0: 0.0, 1: np.nan}, 1: {0: np.nan, 1: 0.0}}
        test_kl_divs = {
            k1: {k2: (v2.item() if not math.isnan(v2.item()) else np.nan) for k2, v2 in v1.items()}
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
        # KL(P(Y | A = 0, C = 1) || P(Y | C = 1)) = NaN
        # KL(P(Y | A = 1, C = 1) || P(Y | C = 1)) = NaN

        # KL divergence cannot be negative
        expected_kl_divs = {0: {0: 0.4055, 1: 1.0986}, 1: {0: np.nan, 1: np.nan}}
        test_kl_divs = {
            k1: {k2: (v2.item() if not math.isnan(v2.item()) else np.nan) for k2, v2 in v1.items()}
            for k1, v1 in self.sufficiency(C, Y, A).items()
        }
        assert len(expected_kl_divs) == len(test_kl_divs)
        assert expected_kl_divs[0] == pytest.approx(test_kl_divs[0], abs=1e-3)
        assert expected_kl_divs[1] == test_kl_divs[1]


class DemographicParityWithoutGroundTruthTest(AllenNlpTestCase):
    def test_invalid_dimensions(self):
        ova_npmixy = DemographicParityWithoutGroundTruth()
        Y = torch.eye(3).long()
        X = torch.eye(4).long()
        with pytest.raises(ConfigurationError):
            ova_npmixy(Y, X)

    def test_invalid_num_classes(self):
        ova_npmixy = DemographicParityWithoutGroundTruth()
        Y = torch.eye(3).long()
        X = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            ova_npmixy(Y, X, 1)

    def test_pmi(self):
        ova_pmi = DemographicParityWithoutGroundTruth("pmi", "ova")
        pairwise_pmi = DemographicParityWithoutGroundTruth("pmi", "pairwise")
        Y = torch.ones(3, 3).long()
        X = torch.eye(3).long()

        # P(X = 0, Y = 0) = 0
        # P(X = 0, Y = 1) = 2/3
        # P(X = 1, Y = 0) = 0
        # P(X = 1, Y = 1) = 1/3
        # P(X = 0) = 2/3
        # P(X = 1) = 1/3
        # P(Y = 0) = 0
        # P(Y = 1) = 1
        # G(Y = 0 | X = 0, X = rest, PMI) = NaN
        # G(Y = 1 | X = 0, X = rest, PMI) = ln(1) - ln(1) = 0.0
        # G(Y = 0 | X = 1, X = rest, PMI) = NaN
        # G(Y = 1 | X = 1, X = rest, PMI) = ln(1) - ln(1) = 0.0
        expected_ova_pmi_gaps = {
            0: [np.nan, 0.0],
            1: [np.nan, 0.0],
        }
        test_ova_pmi_gaps = {
            k: [(e if not math.isnan(e) else np.nan) for e in v.tolist()]
            for k, v in ova_pmi(Y, X).items()
        }
        assert expected_ova_pmi_gaps == test_ova_pmi_gaps

        expected_pairwise_pmi_gaps = {
            0: {0: [np.nan, 0.0], 1: [np.nan, 0.0]},
            1: {0: [np.nan, 0.0], 1: [np.nan, 0.0]},
        }
        test_pairwise_pmi_gaps = {
            k1: {
                k2: [(e if not math.isnan(e) else np.nan) for e in v2.tolist()]
                for k2, v2 in v1.items()
            }
            for k1, v1 in pairwise_pmi(Y, X).items()
        }
        assert expected_pairwise_pmi_gaps == test_pairwise_pmi_gaps

    def test_pmisq(self):
        ova_pmisq = DemographicParityWithoutGroundTruth("pmisq", "ova")
        pairwise_pmisq = DemographicParityWithoutGroundTruth("pmisq", "pairwise")
        Y = torch.ones(3, 3).long()
        X = torch.eye(3).long()

        expected_ova_pmisq_gaps = {
            0: [np.nan, round(math.log(2), 3)],
            1: [np.nan, round(math.log(0.5), 3)],
        }
        test_ova_pmisq_gaps = {
            k: [(round(e, 3) if not math.isnan(e) else np.nan) for e in v.tolist()]
            for k, v in ova_pmisq(Y, X).items()
        }
        assert expected_ova_pmisq_gaps == test_ova_pmisq_gaps

        expected_pairwise_pmisq_gaps = {
            0: {0: [np.nan, 0.0], 1: [np.nan, round(math.log(2), 3)]},
            1: {0: [np.nan, round(math.log(0.5), 3)], 1: [np.nan, 0.0]},
        }
        test_pairwise_pmisq_gaps = {
            k1: {
                k2: [(round(e, 3) if not math.isnan(e) else np.nan) for e in v2.tolist()]
                for k2, v2 in v1.items()
            }
            for k1, v1 in pairwise_pmisq(Y, X).items()
        }
        assert expected_pairwise_pmisq_gaps == test_pairwise_pmisq_gaps

    def test_npmiy(self):
        ova_npmiy = DemographicParityWithoutGroundTruth("npmiy", "ova")
        pairwise_npmiy = DemographicParityWithoutGroundTruth("npmiy", "pairwise")
        Y = torch.ones(3, 3).long()
        X = torch.eye(3).long()

        expected_ova_npmiy_gaps = {
            0: [np.nan, np.nan],
            1: [np.nan, np.nan],
        }
        test_ova_npmiy_gaps = {
            k: [(e if not math.isnan(e) else np.nan) for e in v.tolist()]
            for k, v in ova_npmiy(Y, X).items()
        }
        assert expected_ova_npmiy_gaps == test_ova_npmiy_gaps

        expected_pairwise_npmiy_gaps = {
            0: {0: [np.nan, np.nan], 1: [np.nan, np.nan]},
            1: {0: [np.nan, np.nan], 1: [np.nan, np.nan]},
        }
        test_pairwise_npmiy_gaps = {
            k1: {
                k2: [(e if not math.isnan(e) else np.nan) for e in v2.tolist()]
                for k2, v2 in v1.items()
            }
            for k1, v1 in pairwise_npmiy(Y, X).items()
        }
        assert expected_pairwise_npmiy_gaps == test_pairwise_npmiy_gaps

    def test_npmixy(self):
        ova_npmixy = DemographicParityWithoutGroundTruth("npmixy", "ova")
        pairwise_npmixy = DemographicParityWithoutGroundTruth("npmixy", "pairwise")
        Y = torch.ones(3, 3).long()
        X = torch.eye(3).long()

        expected_ova_npmixy_gaps = {
            0: [np.nan, 0.0],
            1: [np.nan, 0.0],
        }
        test_ova_npmixy_gaps = {
            k: [(e if not math.isnan(e) else np.nan) for e in v.tolist()]
            for k, v in ova_npmixy(Y, X).items()
        }
        assert expected_ova_npmixy_gaps == test_ova_npmixy_gaps

        expected_pairwise_npmixy_gaps = {
            0: {0: [np.nan, 0.0], 1: [np.nan, 0.0]},
            1: {0: [np.nan, 0.0], 1: [np.nan, 0.0]},
        }
        test_pairwise_npmixy_gaps = {
            k1: {
                k2: [(e if not math.isnan(e) else np.nan) for e in v2.tolist()]
                for k2, v2 in v1.items()
            }
            for k1, v1 in pairwise_npmixy(Y, X).items()
        }
        assert expected_pairwise_npmixy_gaps == test_pairwise_npmixy_gaps
