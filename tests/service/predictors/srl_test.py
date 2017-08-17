# pylint: disable=no-self-use,invalid-name
import json
from unittest import TestCase

from allennlp.common import Params
from allennlp.common.params import replace_none
from allennlp.service.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor


class TestSrlPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "sentence": "The squirrel wrote a unit test to make sure its nuts worked as designed."
        }

        with open('tests/fixtures/srl/experiment.json') as f:
            config = json.loads(f.read())
            srl_config = Params(replace_none(config))

        model = SemanticRoleLabelerPredictor.from_config(srl_config)

        result = model.predict_json(inputs)

        # TODO(joelgrus): update this when you figure out the result format
        verbs = result.get("verbs")

        assert verbs is not None

        assert any(v["verb"] == "wrote" for v in verbs)
        assert any(v["verb"] == "make" for v in verbs)
        assert any(v["verb"] == "worked" for v in verbs)
