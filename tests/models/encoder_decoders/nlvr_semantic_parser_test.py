from allennlp.common.testing import ModelTestCase


class NlvrSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(NlvrSemanticParserTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/nlvr_semantic_parser/experiment.json",
                          "tests/fixtures/data/nlvr/sample_data.jsonl")

    def test_model_can_train_svave_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
