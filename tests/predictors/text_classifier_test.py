import math

from pytest import approx

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestTextClassifierPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
            "sentence": "It was the ending that I hated. I was disappointed that it was so bad."
        }

        archive = load_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "text_classifier")
        result = predictor.predict_json(inputs)

        logits = result.get("logits")
        assert logits is not None
        assert isinstance(logits, list)
        assert len(logits) == 2
        assert all(isinstance(x, float) for x in logits)

        probs = result.get("probs")
        assert probs is not None
        assert isinstance(probs, list)
        assert len(probs) == 2
        assert all(isinstance(x, float) for x in probs)
        assert all(x >= 0 for x in probs)
        assert sum(probs) == approx(1.0)

        label = result.get("label")
        assert label is not None
        assert label in predictor._model.vocab.get_token_to_index_vocabulary(namespace="labels")

        exps = [math.exp(x) for x in logits]
        sum_exps = sum(exps)
        for e, p in zip(exps, probs):
            assert e / sum_exps == approx(p)

    def test_batch_prediction(self):
        batch_inputs = [
            {"sentence": "It was the ending that I hated. I was disappointed that it was so bad."},
            {"sentence": "This one is honestly the worst movie I've ever watched."},
        ]

        archive = load_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "text_classifier")
        results = predictor.predict_batch_json(batch_inputs)
        assert len(results) == 2

        for result in results:
            logits = result.get("logits")
            assert logits is not None
            assert isinstance(logits, list)
            assert len(logits) == 2
            assert all(isinstance(x, float) for x in logits)

            probs = result.get("probs")
            assert probs is not None
            assert isinstance(probs, list)
            assert len(probs) == 2
            assert all(isinstance(x, float) for x in probs)
            assert all(x >= 0 for x in probs)
            assert sum(probs) == approx(1.0)

            label = result.get("label")
            assert label is not None
            assert label in predictor._model.vocab.get_token_to_index_vocabulary(namespace="labels")

            exps = [math.exp(x) for x in logits]
            sum_exps = sum(exps)
            for e, p in zip(exps, probs):
                assert e / sum_exps == approx(p)

    def test_predictions_to_labeled_instances(self):
        inputs = {
            "sentence": "It was the ending that I hated. I was disappointed that it was so bad."
        }

        archive = load_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "text_classifier")

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert "label" in new_instances[0].fields
        assert new_instances[0].fields["label"] is not None
        assert len(new_instances) == 1
