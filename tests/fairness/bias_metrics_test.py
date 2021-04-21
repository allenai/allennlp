import pytest
import torch
import json

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.fairness.bias_metrics import (
    WordEmbeddingAssociationTest,
    EmbeddingCoherenceTest,
    NaturalLanguageInference,
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
    def test_invalid_dims(self):
        EmbeddingCoherenceTest()


class NaturalLanguageInferenceTest(AllenNlpTestCase):
    def test_invalid_dims(self):
        NaturalLanguageInference()
