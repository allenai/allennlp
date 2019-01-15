from allennlp.common.testing import ModelTestCase

class TestLanguageModelWithMultiprocessDatasetReader(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'language_model' / 'experiment_multiprocessing_reader.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences*')

    def test_unidirectional_language_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
