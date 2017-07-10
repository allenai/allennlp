# pylint: disable=no-self-use,invalid-name
import codecs

from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.testing.test_case import AllenNlpTestCase


class TestLanguageModellingDatasetReader(AllenNlpTestCase):
    def setUp(self):
        super(TestLanguageModellingDatasetReader, self).setUp()
        self.write_sentence_data()

    def write_sentence_data(self):
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write("This is a sentence for language modelling.\n")
            train_file.write("Here's another one for language modelling.\n")

    def test_read_from_file(self):
        reader = LanguageModelingReader(self.TRAIN_FILE,
                                        tokens_per_instance=4)

        dataset = reader.read()
        instances = dataset.instances
        assert instances[0].fields()["input_tokens"].tokens() == ["<S>", "this", "is", "a", "sentence"]
        assert instances[1].fields()["input_tokens"].tokens() == ["<S>", "for", "language", "modelling", "."]
        assert instances[2].fields()["input_tokens"].tokens() == ["<S>", "here", "'s", "another", "one"]
