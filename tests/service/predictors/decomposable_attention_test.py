# pylint: disable=no-self-use,invalid-name
from unittest import TestCase
import math

from pytest import approx

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class TestDecomposableAttentionPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        archive = load_archive('tests/fixtures/decomposable_attention/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'textual-entailment')
        result = predictor.predict_json(inputs)

        # Label probs should be 3 floats that sum to one
        label_probs = result.get("label_probs")
        assert label_probs is not None
        assert isinstance(label_probs, list)
        assert len(label_probs) == 3
        assert all(isinstance(x, float) for x in label_probs)
        assert all(x >= 0 for x in label_probs)
        assert sum(label_probs) == approx(1.0)

        # Logits should be 3 floats that softmax to label_probs
        label_logits = result.get("label_logits")
        assert label_logits is not None
        assert isinstance(label_logits, list)
        assert len(label_logits) == 3
        assert all(isinstance(x, float) for x in label_logits)

        exps = [math.exp(x) for x in label_logits]
        sumexps = sum(exps)
        for e, p in zip(exps, label_probs):
            assert e / sumexps == approx(p)

    def test_batch_prediction(self):
        batch_inputs = [
                {
                        "premise": "I always write unit tests for my code.",
                        "hypothesis": "One time I didn't write any unit tests for my code."
                },
                {
                        "premise": "I also write batched unit tests for throughput!",
                        "hypothesis": "Batch tests are slower."
                },
        ]

        archive = load_archive('tests/fixtures/decomposable_attention/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'textual-entailment')
        result = predictor.predict_batch_json(batch_inputs)
        # Logits should be 3 floats that softmax to label_probs
        batch_label_logits = result.get("label_logits")
        # Label probs should be 3 floats that sum to one
        batch_label_probs = result.get("label_probs")

        for label_probs, label_logits in zip(batch_label_probs, batch_label_logits):
            assert label_probs is not None
            assert isinstance(label_probs, list)
            assert len(label_probs) == 3
            assert all(isinstance(x, float) for x in label_probs)
            assert all(x >= 0 for x in label_probs)
            assert sum(label_probs) == approx(1.0)

            assert label_logits is not None
            assert isinstance(label_logits, list)
            assert len(label_logits) == 3
            assert all(isinstance(x, float) for x in label_logits)

            exps = [math.exp(x) for x in label_logits]
            sumexps = sum(exps)
            for e, p in zip(exps, label_probs):
                assert e / sumexps == approx(p)
