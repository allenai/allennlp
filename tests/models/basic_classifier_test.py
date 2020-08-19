import numpy

from allennlp.common.testing import ModelTestCase


class TestBasicClassifier(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            self.FIXTURES_ROOT / "basic_classifier" / "experiment_seq2vec.jsonnet",
            self.FIXTURES_ROOT / "data" / "text_classification_json" / "imdb_corpus.jsonl",
        )

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "label" in output_dict.keys()
        probs = output_dict["probs"][0].data.numpy()
        numpy.testing.assert_almost_equal(numpy.sum(probs, -1), numpy.array([1]))

    def test_seq2vec_clf_can_train_save_and_load(self):
        self.set_up_model(
            self.FIXTURES_ROOT / "basic_classifier" / "experiment_seq2vec.jsonnet",
            self.FIXTURES_ROOT / "data" / "text_classification_json" / "imdb_corpus.jsonl",
        )
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_seq2seq_clf_can_train_save_and_load(self):
        self.set_up_model(
            self.FIXTURES_ROOT / "basic_classifier" / "experiment_seq2seq.jsonnet",
            self.FIXTURES_ROOT / "data" / "text_classification_json" / "imdb_corpus.jsonl",
        )
        self.ensure_model_can_train_save_and_load(self.param_file)
