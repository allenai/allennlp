from allennlp.common.testing import ModelTestCase, AllenNlpTestCase


class TestBasicClassifier(ModelTestCase):

    def test_seq2vec_clf_can_train_save_and_load(self):
        data_directory = AllenNlpTestCase.FIXTURES_ROOT / "data" / "text_classification_json"
        self.set_up_model(AllenNlpTestCase.FIXTURES_ROOT / 'basic_classifier' / 'experiment_seq2vec.jsonnet',
                          data_directory / "imdb_corpus.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_seq2seq_clf_can_train_save_and_load(self):
        data_directory = AllenNlpTestCase.FIXTURES_ROOT / "data" / "text_classification_json"
        self.set_up_model(AllenNlpTestCase.FIXTURES_ROOT / 'basic_classifier' / 'experiment_seq2seq.jsonnet',
                          data_directory / "imdb_corpus.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)
