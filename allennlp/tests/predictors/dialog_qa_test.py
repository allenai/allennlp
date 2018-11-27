# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestDialogQAPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {"paragraphs": [{"qas": [{"followup": "y", "yesno": "x", "question": "When was the first one?",
                                           "answers": [{"answer_start": 0, "text": "One time"}], "id": "C_q#0"},
                                          {"followup": "n", "yesno": "x", "question": "What were you doing?",
                                           "answers": [{"answer_start": 15, "text": "writing a"}], "id": "C_q#1"},
                                          {"followup": "m", "yesno": "y", "question": "How often?",
                                           "answers": [{"answer_start": 4, "text": "time I"}], "id": "C_q#2"}],
                                  "context": "One time I was writing a unit test,\
                                   and it succeeded on the first attempt."}]}

        archive = load_archive(self.FIXTURES_ROOT / 'dialog_qa' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'dialog_qa')

        result = predictor.predict_json(inputs)

        best_span_str_list = result.get("best_span_str")
        for best_span_str in best_span_str_list:
            assert isinstance(best_span_str, str)
            assert best_span_str != ""

    def test_batch_prediction(self):
        inputs = [{"paragraphs": [{"qas": [{"followup": "y", "yesno": "x", "question": "When was the first one?",
                                            "answers": [{"answer_start": 0, "text": "One time"}], "id": "C_q#0"},
                                           {"followup": "n", "yesno": "x", "question": "What were you doing?",
                                            "answers": [{"answer_start": 15, "text": "writing a"}], "id": "C_q#1"},
                                           {"followup": "m", "yesno": "y", "question": "How often?",
                                            "answers": [{"answer_start": 4, "text": "time I"}], "id": "C_q#2"}],
                                   "context": "One time I was writing a unit test,\
                                    and it succeeded on the first attempt."}]},
                  {"paragraphs": [{"qas": [{"followup": "y", "yesno": "x", "question": "When was the first one?",
                                            "answers": [{"answer_start": 0, "text": "One time"}], "id": "C_q#0"},
                                           {"followup": "n", "yesno": "x", "question": "What were you doing?",
                                            "answers": [{"answer_start": 15, "text": "writing a"}], "id": "C_q#1"},
                                           {"followup": "m", "yesno": "y", "question": "How often?",
                                            "answers": [{"answer_start": 4, "text": "time I"}], "id": "C_q#2"}],
                                   "context": "One time I was writing a unit test,\
                                    and it succeeded on the first attempt."}]}]

        archive = load_archive(self.FIXTURES_ROOT / 'dialog_qa' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'dialog_qa')

        results = predictor.predict_batch_json(inputs)
        assert len(results) == 2
