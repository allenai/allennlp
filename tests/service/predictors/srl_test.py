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
        predictor = Predictor.from_archive(archive)

        result = predictor.predict_json(inputs)

        # TODO(joelgrus): update this when you figure out the result format
        verbs = result.get("verbs")

        assert verbs is not None

        assert any(v["verb"] == "wrote" for v in verbs)
        assert any(v["verb"] == "make" for v in verbs)
        assert any(v["verb"] == "worked" for v in verbs)
