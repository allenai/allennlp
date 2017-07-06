# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.testing.test_case import AllenNlpTestCase


class TestLanguageModellingDatasetReader(AllenNlpTestCase):

    def setUp(self):
        super(TestLanguageModellingDatasetReader, self).setUp()
        self.write_sentence_data()

    def test_read_from_file(self):
        reader = LanguageModelingReader(tokens_per_instance=4)

        dataset = reader.read(self.TRAIN_FILE)
        instances = dataset.instances
        assert instances[0].fields()["input_tokens"].tokens() == ["<S>", "this", "is", "a", "sentence"]
        assert instances[1].fields()["input_tokens"].tokens() == ["<S>", "for", "language", "modelling", "."]
        assert instances[2].fields()["input_tokens"].tokens() == ["<S>", "here", "'s", "another", "one"]
