from allennlp.common.testing.model_test_case import ModelTestCase


class BiaffineDependencyParserMultilangTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "biaffine_dependency_parser_multilang" / "experiment.json",
            self.FIXTURES_ROOT / "data" / "dependencies_multilang" / "*",
        )

    def test_dependency_parser_can_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.param_file, gradients_to_ignore={"arc_attention._bias"}
        )

    def test_mst_decoding_can_run_forward(self):
        self.model.use_mst_decoding_for_validation = True
        self.ensure_model_can_train_save_and_load(
            self.param_file, gradients_to_ignore={"arc_attention._bias"}
        )
