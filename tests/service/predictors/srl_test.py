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

    def test_prediction_with_no_verbs(self):

        input1 = {"sentence": "Blah no verb sentence."}
        archive = load_archive('tests/fixtures/srl/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'semantic-role-labeling')
        result = predictor.predict_json(input1)
        assert result == {'words': ['Blah', 'no', 'verb', 'sentence', '.'], 'verbs': []}

        input2 = {"sentence": "This sentence has a verb."}
        results = predictor.predict_batch_json([input1, input2])
        assert results[0] == {'words': ['Blah', 'no', 'verb', 'sentence', '.'], 'verbs': []}
        assert results[1] == {'words': ['This', 'sentence', 'has', 'a', 'verb', '.'],
                              'verbs': [{'verb': 'has', 'description': 'This sentence has a verb .',
                                         'tags': ['O', 'O', 'O', 'O', 'O', 'O']}]}
