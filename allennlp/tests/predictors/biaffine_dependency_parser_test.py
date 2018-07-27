# pylint: disable=no-self-use,invalid-name
from pytest import approx

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestBiaffineDependencyParser(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "sentence": "Please could you parse this sentence?",
        }

        archive = load_archive(self.FIXTURES_ROOT / 'biaffine_dependency_parser' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'biaffine-dependency-parser')

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
                        "sentence": "What kind of test succeeded on its first attempt?",
                },
                {
                        "sentence": "What kind of test succeeded on its first attempt at batch processing?",
                }
        ]

        archive = load_archive(self.FIXTURES_ROOT / 'biaffine_dependency_parser' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'biaffine-dependency-parser')

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
