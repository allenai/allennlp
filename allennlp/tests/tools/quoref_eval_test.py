from allennlp.common.testing import AllenNlpTestCase
from allennlp.tools import quoref_eval


class TestQuorefEval(AllenNlpTestCase):
    def test_quoref_eval(self):
        predictions_file = self.FIXTURES_ROOT / "data" / "quoref" / "quoref_sample_predictions.json"
        gold_file = self.FIXTURES_ROOT / "data" / "quoref" / "quoref_sample.json"
        metrics = quoref_eval.evaluate_prediction_file(predictions_file, gold_file)
        assert metrics == (0.5, 0.625)
