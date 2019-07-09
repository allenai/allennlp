# pylint: disable=no-self-use,invalid-name
import math

from pytest import approx

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestDecomposableAttentionPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        archive = load_archive(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
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

        archive = load_archive(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'textual-entailment')
        results = predictor.predict_batch_json(batch_inputs)
        print(results)
        assert len(results) == 2

        for result in results:
            # Logits should be 3 floats that softmax to label_probs
            label_logits = result.get("label_logits")
            # Label probs should be 3 floats that sum to one
            label_probs = result.get("label_probs")
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

    def test_predictions_to_labeled_instances(self):
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        archive = load_archive(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'textual-entailment')

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert 'hypothesis' in new_instances[0].fields
        assert 'premise' in new_instances[0].fields
        assert new_instances[0].fields['hypothesis'] is not None
        assert new_instances[0].fields['premise'] is not None

    def test_get_gradients(self):
        inputs = {
                "premise": "I always write unit tests",
                "hypothesis": "One time I did not write any unit tests"
        }

        archive = load_archive(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'textual-entailment')

        labeled_instances = predictor.inputs_to_labeled_instances(inputs)
        for idx, instance in enumerate(labeled_instances):
            grads = predictor.get_gradients([instance])[0]
            assert 'grad_input_1' in grads
            assert 'grad_input_2' in grads
            assert grads['grad_input_1'] is not None
            assert grads['grad_input_2'] is not None
            assert len(grads['grad_input_1']) == 9  # 9 words in hypothesis
            assert len(grads['grad_input_2']) == 5  # 5 words in premise
