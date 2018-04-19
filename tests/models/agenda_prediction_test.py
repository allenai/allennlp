from allennlp.common.testing import ModelTestCase


class AgendaPredictorTest(ModelTestCase):
    def setUp(self):
        super(AgendaPredictorTest, self).setUp()
        self.set_up_model("tests/fixtures/agenda_predictor/experiment.json",
                          "tests/fixtures/data/nlvr/sample_processed_data.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
