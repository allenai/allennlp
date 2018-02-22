# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class TestConstituencyParserPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "sentence": "What a great test sentence.",
        }

        archive = load_archive('tests/fixtures/constituency_parser/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'constituency-parser')
        result = predictor.predict_json(inputs)

        assert len(result["spans"]) == 21
        assert result["sentence"] == ["What", "a", "great", "test", "sentence", "."]
        assert isinstance(result["trees"], str)

        for class_distribution in result["class_probabilities"]:
            self.assertAlmostEqual(sum(class_distribution), 1.0, places=4)

    def test_batch_prediction(self):
        inputs = [
                {"sentence": "What a great test sentence."},
                {"sentence": "Here's another good, interesting one."}
        ]

        archive = load_archive('tests/fixtures/constituency_parser/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'constituency-parser')
        results = predictor.predict_batch_json(inputs)

        result = results[0]
        assert len(result["spans"]) == 21
        assert len(result["class_probabilities"]) == 21
        assert result["sentence"] == ["What", "a", "great", "test", "sentence", "."]
        assert isinstance(result["trees"], str)

        for class_distribution in result["class_probabilities"]:
            self.assertAlmostEqual(sum(class_distribution), 1.0, places=4)
        
        result = results[1]

        assert len(result["spans"]) == 36
        assert len(result["class_probabilities"]) == 36
        assert result["sentence"] == ["Here", "'s", "another", "good", ",", "interesting", "one", "."]
        assert isinstance(result["trees"], str)

        for class_distribution in result["class_probabilities"]:
            self.assertAlmostEqual(sum(class_distribution), 1.0, places=4)
