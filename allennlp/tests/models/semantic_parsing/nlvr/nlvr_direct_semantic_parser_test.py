from allennlp.common.testing import ModelTestCase


class NlvrDirectSemanticParserTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT
            / "semantic_parsing"
            / "nlvr_direct_semantic_parser"
            / "experiment.json",
            self.FIXTURES_ROOT / "data" / "nlvr" / "sample_processed_data.jsonl",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
