# pylint: disable=no-self-use,invalid-name
import os
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

import pytest

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


    @pytest.mark.skipif(os.environ.get("TRAVIS") is not None, reason="causes OOM error and crashes on Travis")
    def test_cpu_vs_gpu(self):
        gpu_archive = load_archive('tests/fixtures/srl/serialization/model_gpu.tar.gz')
        predictor_gpu = Predictor.from_archive(gpu_archive)

        cpu_archive = load_archive('tests/fixtures/srl/serialization/model.tar.gz')
        predictor_cpu = Predictor.from_archive(cpu_archive)

        sentences = [
                "Squirrels write unit tests to make sure their nuts work correctly.",
                "My code never works when I need it to.",
                "An extra GPU could really help my experiments run faster."
        ]

        for sentence in sentences:
            prediction_cpu = predictor_cpu.predict_json({"sentence": sentence})
            prediction_gpu = predictor_gpu.predict_json({"sentence": sentence})

            assert set(prediction_cpu.keys()) == {"verbs"}
            assert set(prediction_gpu.keys()) == {"verbs"}

            for cverb, gverb in zip(prediction_cpu["verbs"], prediction_gpu["verbs"]):
                assert cverb["index"] == gverb["index"]
                assert cverb["verb"] == gverb["verb"]
                assert cverb["tags"] == gverb["tags"]
                assert cverb["class_probabilities"] == pytest.approx(gverb["class_probabilities"])
