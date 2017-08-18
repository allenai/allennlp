# pylint: disable=no-self-use,invalid-name
import os
from unittest import TestCase

from allennlp.common import Params
from allennlp.models import Model
from allennlp.service.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor

import pytest

class TestSrlPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "sentence": "The squirrel wrote a unit test to make sure its nuts worked as designed."
        }

        config = Params.from_file('tests/fixtures/srl/experiment.json')
        predictor = SemanticRoleLabelerPredictor.from_config(config)

        result = predictor.predict_json(inputs)

        # TODO(joelgrus): update this when you figure out the result format
        verbs = result.get("verbs")

        assert verbs is not None

        assert any(v["verb"] == "wrote" for v in verbs)
        assert any(v["verb"] == "make" for v in verbs)
        assert any(v["verb"] == "worked" for v in verbs)


    @pytest.mark.skipif(os.environ.get("TRAVIS") is not None, reason="causes OOM error and crashes on Travis")
    def test_cpu_vs_gpu(self):
        config = Params.from_file('tests/fixtures/srl/experiment.json')
        predictor_gpu = SemanticRoleLabelerPredictor.from_config(config)

        # params have been consumed, so reload them
        config = Params.from_file('tests/fixtures/srl/experiment.json')
        model_cpu = Model.load(config, weights_file='tests/fixtures/srl/serialization/best_cpu.th')

        predictor_cpu = SemanticRoleLabelerPredictor(
                model=model_cpu,
                tokenizer=predictor_gpu.tokenizer,
                token_indexers=predictor_gpu.token_indexers
        )

        sentences = [
                "Squirrels write unit tests to make sure their nuts work correctly.",
                "My code never works when I need it to.",
                "An extra GPU could really help my experiments run faster."
        ]

        for sentence in sentences:
            prediction_cpu = predictor_cpu.predict_json({"sentence": sentence})
            prediction_gpu = predictor_gpu.predict_json({"sentence": sentence})
            print(prediction_cpu)
            assert prediction_cpu == prediction_gpu
