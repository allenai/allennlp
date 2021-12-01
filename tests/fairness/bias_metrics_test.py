import pytest
import torch
import json
import math
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    global_distributed_metric,
    run_distributed_test,
)
from allennlp.fairness.bias_metrics import (
    WordEmbeddingAssociationTest,
    EmbeddingCoherenceTest,
    NaturalLanguageInference,
    AssociationWithoutGroundTruth,
)


class WordEmbeddingAssociationTestTest(AllenNlpTestCase):
    def setup_method(self):
        # embedding data from VERB demo
        emb_filename = str(self.FIXTURES_ROOT / "fairness" / "bias_embeddings.json")
        with open(emb_filename) as emb_file:
            emb_data = json.load(emb_file)

        self.X = torch.cat(
            [
                torch.Tensor(emb_data["he"]).reshape(1, -1),
                torch.Tensor(emb_data["him"]).reshape(1, -1),
            ]
        )
        self.Y = torch.cat(
            [
                torch.Tensor(emb_data["she"]).reshape(1, -1),
                torch.Tensor(emb_data["her"]).reshape(1, -1),
            ]
        )
        self.A = torch.cat(
            [
                torch.Tensor(emb_data["engineer"]).reshape(1, -1),
                torch.Tensor(emb_data["banker"]).reshape(1, -1),
            ]
        )
        self.B = torch.cat(
            [
                torch.Tensor(emb_data["nurse"]).reshape(1, -1),
                torch.Tensor(emb_data["receptionist"]).reshape(1, -1),
            ]
        )

    def teardown_method(self):
        pass

    def test_invalid_dims(self):
        weat = WordEmbeddingAssociationTest()
        with pytest.raises(ConfigurationError):
            weat(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2))
        with pytest.raises(ConfigurationError):
            weat(torch.zeros((2, 2)), torch.zeros((2, 2)), torch.zeros(2), torch.zeros(2))
        with pytest.raises(ConfigurationError):
            weat(torch.zeros((2, 2)), torch.zeros((2, 3)), torch.zeros((2, 2)), torch.zeros((2, 2)))
        with pytest.raises(ConfigurationError):
            weat(torch.zeros((2, 2)), torch.zeros((2, 2)), torch.zeros((2, 3)), torch.zeros((2, 2)))

    @multi_device
    def test_weat(self, device: str):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.A = self.A.to(device)
        self.B = self.B.to(device)

        weat = WordEmbeddingAssociationTest()
        test_weat_score = weat(self.X, self.Y, self.A, self.B)
        assert test_weat_score.item() == pytest.approx(1.872, rel=1e-4)


class EmbeddingCoherenceTestTest(AllenNlpTestCase):
    def setup_method(self):
        # embedding data from VERB demo
        emb_filename = str(self.FIXTURES_ROOT / "fairness" / "bias_embeddings.json")
        with open(emb_filename) as emb_file:
            emb_data = json.load(emb_file)

        self.X = torch.cat(
            [
                torch.Tensor(emb_data["he"]).reshape(1, -1),
                torch.Tensor(emb_data["him"]).reshape(1, -1),
            ]
        )
        self.Y = torch.cat(
            [
                torch.Tensor(emb_data["she"]).reshape(1, -1),
                torch.Tensor(emb_data["her"]).reshape(1, -1),
            ]
        )
        self.AB = torch.cat(
            [
                torch.Tensor(emb_data["engineer"]).reshape(1, -1),
                torch.Tensor(emb_data["banker"]).reshape(1, -1),
                torch.Tensor(emb_data["nurse"]).reshape(1, -1),
                torch.Tensor(emb_data["receptionist"]).reshape(1, -1),
            ]
        )

    def teardown_method(self):
        pass

    def test_invalid_dims(self):
        ect = EmbeddingCoherenceTest()
        with pytest.raises(ConfigurationError):
            ect(torch.zeros(2), torch.zeros(2), torch.zeros(2))
        with pytest.raises(ConfigurationError):
            ect(torch.zeros((2, 2)), torch.zeros((2, 2)), torch.zeros(2))
        with pytest.raises(ConfigurationError):
            ect(torch.zeros((2, 2)), torch.zeros((2, 3)), torch.zeros((2, 2)))
        with pytest.raises(ConfigurationError):
            ect(torch.zeros((2, 2)), torch.zeros((2, 2)), torch.zeros((2, 3)))

    @multi_device
    def test_ect(self, device: str):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.AB = self.AB.to(device)

        ect = EmbeddingCoherenceTest()
        test_ect_score = ect(self.X, self.Y, self.AB)
        assert test_ect_score.item() == pytest.approx(0.800, rel=1e-4)


class NaturalLanguageInferenceTest(AllenNlpTestCase):
    def test_invalid_dimensions(self):
        nli_probabilities = torch.ones(3)
        with pytest.raises(ConfigurationError):
            NaturalLanguageInference(0)(nli_probabilities)

        nli_probabilities = torch.eye(4)
        with pytest.raises(ConfigurationError):
            NaturalLanguageInference(0)(nli_probabilities)

    @multi_device
    def test_nli(self, device: str):
        nli_probabilities = 0.6 * torch.eye(3, device=device)
        nli = NaturalLanguageInference(0)
        nli(nli_probabilities)

        expected_scores = {
            "net_neutral": 0.6 / 3,
            "fraction_neutral": 1 / 3,
            "threshold_0.5": 1 / 3,
            "threshold_0.7": 0.0,
        }
        assert nli.get_metric(reset=True) == pytest.approx(expected_scores)
        assert all([v == 0.0 for k, v in nli.get_metric().items()])

    def test_distributed_nli(self):
        nli_probabilities = 0.6 * torch.eye(3)
        expected_scores = {
            "net_neutral": 0.6 / 3,
            "fraction_neutral": 1 / 3,
            "threshold_0.5": 1 / 3,
            "threshold_0.7": 0.0,
        }
        metric_kwargs = {"nli_probabilities": [nli_probabilities, nli_probabilities]}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            NaturalLanguageInference(0),
            metric_kwargs,
            expected_scores,
            exact=False,
        )


class AssociationWithoutGroundTruthTest(AllenNlpTestCase):
    def test_invalid_dimensions(self):
        ova_npmixy = AssociationWithoutGroundTruth(2, 2)
        Y = torch.eye(3).long()
        X = torch.eye(4).long()
        with pytest.raises(ConfigurationError):
            ova_npmixy(Y, X)

    def test_invalid_num_classes(self):
        ova_npmixy = AssociationWithoutGroundTruth(1, 1)
        Y = torch.eye(3).long()
        X = torch.eye(3).long()
        with pytest.raises(ConfigurationError):
            ova_npmixy(Y, X)

    @multi_device
    def test_pmi_unmasked_computation(self, device: str):
        ova_pmi = AssociationWithoutGroundTruth(2, 2, "pmi", "ova")
        pairwise_pmi = AssociationWithoutGroundTruth(2, 2, "pmi", "pairwise")
        Y = torch.ones(3, 3, device=device).long()
        X = torch.eye(3, device=device).long()

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

        ova_pmi(Y, X)
        test_ova_pmi_gaps = {
            k: [(e if not math.isnan(e) else np.nan) for e in v.tolist()]  # type: ignore
            for k, v in ova_pmi.get_metric().items()
        }
        assert expected_ova_pmi_gaps == test_ova_pmi_gaps

        ova_pmi(Y, X)
        test_ova_pmi_gaps = {
            k: [(e if not math.isnan(e) else np.nan) for e in v.tolist()]  # type: ignore
            for k, v in ova_pmi.get_metric(reset=True).items()
        }
        assert expected_ova_pmi_gaps == test_ova_pmi_gaps

        test_ova_pmi_gaps = {
            k: [(e if not math.isnan(e) else np.nan) for e in v.tolist()]  # type: ignore
            for k, v in ova_pmi.get_metric(reset=True).items()
        }
        assert test_ova_pmi_gaps == {0: [np.nan, np.nan], 1: [np.nan, np.nan]}

        expected_pairwise_pmi_gaps = {
            0: {0: [np.nan, 0.0], 1: [np.nan, 0.0]},
            1: {0: [np.nan, 0.0], 1: [np.nan, 0.0]},
        }

        pairwise_pmi(Y, X)
        test_pairwise_pmi_gaps = {
            k1: {
                k2: [(e if not math.isnan(e) else np.nan) for e in v2.tolist()]  # type: ignore
                for k2, v2 in v1.items()
            }
            for k1, v1 in pairwise_pmi.get_metric().items()
        }
        assert expected_pairwise_pmi_gaps == test_pairwise_pmi_gaps

        pairwise_pmi(Y, X)
        test_pairwise_pmi_gaps = {
            k1: {
                k2: [(e if not math.isnan(e) else np.nan) for e in v2.tolist()]
                for k2, v2 in v1.items()
            }
            for k1, v1 in pairwise_pmi.get_metric(reset=True).items()
        }
        assert expected_pairwise_pmi_gaps == test_pairwise_pmi_gaps

        test_pairwise_pmi_gaps = {
            k1: {
                k2: [(e if not math.isnan(e) else np.nan) for e in v2.tolist()]
                for k2, v2 in v1.items()
            }
            for k1, v1 in pairwise_pmi.get_metric(reset=True).items()
        }
        assert test_pairwise_pmi_gaps == {
            0: {0: [np.nan, np.nan], 1: [np.nan, np.nan]},
            1: {0: [np.nan, np.nan], 1: [np.nan, np.nan]},
        }

    @multi_device
    def test_pmisq_masked_computation(self, device: str):
        ova_pmisq = AssociationWithoutGroundTruth(2, 2, "pmisq", "ova")
        pairwise_pmisq = AssociationWithoutGroundTruth(2, 2, "pmisq", "pairwise")
        Y = torch.ones(3, 3, device=device).long()
        X = torch.eye(3, device=device).long()
        mask = torch.ones_like(Y).bool()

        expected_ova_pmisq_gaps = {
            0: [np.nan, round(math.log(2), 3)],
            1: [np.nan, round(math.log(0.5), 3)],
        }
        ova_pmisq(Y, X, mask)
        test_ova_pmisq_gaps = {
            k: [(round(e, 3) if not math.isnan(e) else np.nan) for e in v.tolist()]  # type: ignore
            for k, v in ova_pmisq.get_metric().items()
        }
        assert expected_ova_pmisq_gaps == test_ova_pmisq_gaps

        expected_pairwise_pmisq_gaps = {
            0: {0: [np.nan, 0.0], 1: [np.nan, round(math.log(2), 3)]},
            1: {0: [np.nan, round(math.log(0.5), 3)], 1: [np.nan, 0.0]},
        }
        pairwise_pmisq(Y, X, mask)
        test_pairwise_pmisq_gaps = {
            k1: {
                k2: [(round(e, 3) if not math.isnan(e) else np.nan) for e in v2.tolist()]
                for k2, v2 in v1.items()
            }
            for k1, v1 in pairwise_pmisq.get_metric().items()
        }
        assert expected_pairwise_pmisq_gaps == test_pairwise_pmisq_gaps

    def test_distributed_npmiy_unmasked_computation(self):
        Y = torch.ones(3, 3).long()
        X = torch.eye(3).long()

        expected_ova_npmiy_gaps = {
            0: [np.nan, np.nan],
            1: [np.nan, np.nan],
        }
        metric_kwargs = {"predicted_labels": Y, "protected_variable_labels": X}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            AssociationWithoutGroundTruth(2, 2, "npmiy", "ova"),
            metric_kwargs,
            expected_ova_npmiy_gaps,
            exact=True,
        )

        expected_pairwise_npmiy_gaps = {
            0: {0: [np.nan, np.nan], 1: [np.nan, np.nan]},
            1: {0: [np.nan, np.nan], 1: [np.nan, np.nan]},
        }
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            AssociationWithoutGroundTruth(2, 2, "npmiy", "pairwise"),
            metric_kwargs,
            expected_pairwise_npmiy_gaps,
            exact=True,
        )

    def test_distributed_npmixy_masked_computation(self):
        Y = torch.ones(3, 3).long()
        X = torch.eye(3).long()
        mask = torch.ones_like(Y).bool()

        expected_ova_npmixy_gaps = {
            0: [np.nan, 0.0],
            1: [np.nan, 0.0],
        }
        metric_kwargs = {"predicted_labels": Y, "protected_variable_labels": X, "mask": mask}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            AssociationWithoutGroundTruth(2, 2, "npmixy", "ova"),
            metric_kwargs,
            expected_ova_npmixy_gaps,
            exact=True,
        )

        expected_pairwise_npmixy_gaps = {
            0: {0: [np.nan, 0.0], 1: [np.nan, 0.0]},
            1: {0: [np.nan, 0.0], 1: [np.nan, 0.0]},
        }
        metric_kwargs = {"predicted_labels": Y, "protected_variable_labels": X, "mask": mask}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            AssociationWithoutGroundTruth(2, 2, "npmixy", "pairwise"),
            metric_kwargs,
            expected_pairwise_npmixy_gaps,
            exact=True,
        )
