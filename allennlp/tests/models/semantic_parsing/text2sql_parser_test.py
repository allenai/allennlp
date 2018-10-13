from allennlp.common.testing import ModelTestCase


class Text2SqlParserTest(ModelTestCase):

    def setUp(self):
        super().setUp()

        self.set_up_model(str(self.FIXTURES_ROOT / "semantic_parsing" / "text2sql" / "experiment.json"),
                          str(self.FIXTURES_ROOT / "data" / "text2sql" / "restaurants_tiny.json"))

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
