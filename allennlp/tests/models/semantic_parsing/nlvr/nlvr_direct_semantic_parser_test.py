from allennlp.common.testing import ModelTestCase


class NlvrDirectSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(NlvrDirectSemanticParserTest, self).setUp()
        self.set_up_model("tests/fixtures/semantic_parsing/nlvr_direct_semantic_parser/experiment.json",
                          "tests/fixtures/data/nlvr/sample_processed_data.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
