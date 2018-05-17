# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from pytest import approx

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class TestBidafPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        archive = load_archive('tests/fixtures/bidaf/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'machine-comprehension')

        result = predictor.predict_json(inputs)

        best_span = result.get("best_span")
        assert best_span is not None
        assert isinstance(best_span, list)
        assert len(best_span) == 2
        assert all(isinstance(x, int) for x in best_span)
        assert best_span[0] <= best_span[1]

        best_span_str = result.get("best_span_str")
        assert isinstance(best_span_str, str)
        assert best_span_str != ""

        for probs_key in ("span_start_probs", "span_end_probs"):
            probs = result.get(probs_key)
            assert probs is not None
            assert all(isinstance(x, float) for x in probs)
            assert sum(probs) == approx(1.0)

    def test_batch_prediction(self):
        inputs = [
                {
                        "question": "What kind of test succeeded on its first attempt?",
                        "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
                },
                {
                        "question": "What kind of test succeeded on its first attempt at batch processing?",
                        "passage": "One time I was writing a unit test, and it always failed!"
                }
        ]

        archive = load_archive('tests/fixtures/bidaf/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'machine-comprehension')

        results = predictor.predict_batch_json(inputs)
        assert len(results) == 2

        for result in results:
            best_span = result.get("best_span")
            best_span_str = result.get("best_span_str")
            start_probs = result.get("span_start_probs")
            end_probs = result.get("span_end_probs")
            assert best_span is not None
            assert isinstance(best_span, list)
            assert len(best_span) == 2
            assert all(isinstance(x, int) for x in best_span)
            assert best_span[0] <= best_span[1]

            assert isinstance(best_span_str, str)
            assert best_span_str != ""

            for probs in (start_probs, end_probs):
                assert probs is not None
                assert all(isinstance(x, float) for x in probs)
                assert sum(probs) == approx(1.0)
