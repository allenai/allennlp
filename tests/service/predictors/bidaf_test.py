# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class TestBidafPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        archive = load_archive('tests/fixtures/bidaf/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive)

        result = predictor.predict_json(inputs)

        assert "span_start_probs" in result
        assert "span_end_probs" in result
