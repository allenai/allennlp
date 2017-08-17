# pylint: disable=no-self-use,invalid-name
import json
from unittest import TestCase

from allennlp.common import Params
from allennlp.common.params import replace_none
from allennlp.service.predictors.decomposable_attention import DecomposableAttentionPredictor


class TestDecomposableAttentionPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        with open('tests/fixtures/decomposable_attention/experiment.json') as f:
            config = json.loads(f.read())
            decomposable_attention_config = Params(replace_none(config))


        model = DecomposableAttentionPredictor.from_config(decomposable_attention_config)

        result = model.predict_json(inputs)

        assert "label_probs" in result
