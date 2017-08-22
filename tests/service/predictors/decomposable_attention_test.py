# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.common import Params
from allennlp.common.testing.predictors import predictor_from_config
from allennlp.service.predictors.decomposable_attention import DecomposableAttentionPredictor


class TestDecomposableAttentionPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        config = Params.from_file('tests/fixtures/decomposable_attention/experiment.json')
        predictor = predictor_from_config(config)

        result = predictor.predict_json(inputs)

        assert "label_probs" in result
