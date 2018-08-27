# pylint: disable=no-self-use,invalid-name
from pytest import approx

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestDialogQAPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "": [],
                "": [],
                "question": ["When was the first meeting?", "How many people attended it?", "Was the meeting successful?"],
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        archive = load_archive(self.FIXTURES_ROOT / 'dialog_qa' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'dialog_qa')

        result = predictor.predict_json(inputs)

        best_span_str = result.get("best_span_str")
        assert isinstance(best_span_str, str)
        assert best_span_str != ""


    def test_batch_prediction(self):
        inputs = [
                {
                        "question": ["When was the first meeting?", "How many people attended it?", "Was the meeting successful?"],
                        "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
                },
                {
                        "question": ["When was the first meeting?", "How many people attended it?", "Was the meeting successful?"],
                        "passage": "One time I was writing a unit test, and it always failed!"
                }
        ]

        archive = load_archive(self.FIXTURES_ROOT / 'dialog_qa' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'dialog_qa')

        results = predictor.predict_batch_json(inputs)
        assert len(results) == 2

        for result in results:
            best_span_str = result.get("best_span_str")
            assert isinstance(best_span_str, str)
            assert best_span_str != ""
