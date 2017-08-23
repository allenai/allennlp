# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class TestDecomposableAttentionPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        archive = load_archive('tests/fixtures/decomposable_attention/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive)
        result = predictor.predict_json(inputs)

        assert "label_probs" in result
