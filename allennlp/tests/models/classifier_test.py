from allennlp.common.testing import ModelTestCase, AllenNlpTestCase


class TestClassifiers(ModelTestCase):

    def test_linear_clf_can_train_save_and_load(self):
        data_directory = AllenNlpTestCase.FIXTURES_ROOT / "data" / "text_classification_json"
        self.set_up_model(AllenNlpTestCase.FIXTURES_ROOT / 'classifier' / 'experiment_bow_linear.json',
                          data_directory / "imdb_train.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_seq2vec_clf_can_train_save_and_load(self):
        data_directory = AllenNlpTestCase.FIXTURES_ROOT / "data" / "text_classification_json"
        self.set_up_model(AllenNlpTestCase.FIXTURES_ROOT / 'classifier' / 'experiment_seq2vec.json',
                          data_directory / "imdb_train.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_seq2seq_clf_can_train_save_and_load(self):
        data_directory = AllenNlpTestCase.FIXTURES_ROOT / "data" / "text_classification_json"
        self.set_up_model(AllenNlpTestCase.FIXTURES_ROOT / 'classifier' / 'experiment_seq2seq.json',
                          data_directory / "imdb_train.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)
