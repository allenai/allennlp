# pylint: disable=invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.tools import quoref_eval


class TestQuorefEval(AllenNlpTestCase):
    def test_quoref_eval_with_original_data_format(self):
        predictions_file = self.FIXTURES_ROOT / "data" / "quoref" / "quoref_sample_predictions.json"
        gold_file = self.FIXTURES_ROOT / "data" / "quoref" / "quoref_sample.json"
        metrics = quoref_eval.evaluate_prediction_file(predictions_file, gold_file)
        assert metrics == (0.5, 0.625)

    def test_quoref_eval_with_simple_format(self):
        predictions_file = self.FIXTURES_ROOT / "data" / "quoref" / "quoref_sample_predictions.json"
        gold_file = self.FIXTURES_ROOT / "data" / "quoref" / "quoref_sample_predictions.json"
        metrics = quoref_eval.evaluate_prediction_file(predictions_file, gold_file)
        assert metrics == (1.0, 1.0)
