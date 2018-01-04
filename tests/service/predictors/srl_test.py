# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class TestSrlPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "sentence": "The squirrel wrote a unit test to make sure its nuts worked as designed."
        }

        archive = load_archive('tests/fixtures/srl/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'semantic-role-labeling')

        result = predictor.predict_json(inputs)

        words = result.get("words")
        assert words == ["The", "squirrel", "wrote", "a", "unit", "test",
                         "to", "make", "sure", "its", "nuts", "worked", "as", "designed", "."]
        num_words = len(words)

        verbs = result.get("verbs")
        assert verbs is not None
        assert isinstance(verbs, list)

        assert any(v["verb"] == "wrote" for v in verbs)
        assert any(v["verb"] == "make" for v in verbs)
        assert any(v["verb"] == "worked" for v in verbs)

        for verb in verbs:
            tags = verb.get("tags")
            assert tags is not None
            assert isinstance(tags, list)
            assert all(isinstance(tag, str) for tag in tags)
            assert len(tags) == num_words

    def test_batch_prediction(self):
        inputs = {
                "sentence": "The squirrel wrote a unit test to make sure its nuts worked as designed."
        }
        archive = load_archive('tests/fixtures/srl/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'semantic-role-labeling')
        result = predictor.predict_batch_json([inputs, inputs])
        assert result[0] == result[1]
