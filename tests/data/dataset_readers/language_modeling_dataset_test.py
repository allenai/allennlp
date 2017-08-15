# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.common.testing import AllenNlpTestCase


class TestLanguageModellingDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = LanguageModelingReader(tokens_per_instance=4)

        dataset = reader.read('tests/fixtures/data/language_modeling.txt')
        instances = dataset.instances
        assert instances[0].fields()["input_tokens"].tokens() == ["<S>", "This", "is", "a", "sentence"]
        assert instances[1].fields()["input_tokens"].tokens() == ["<S>", "for", "language", "modelling", "."]
        assert instances[2].fields()["input_tokens"].tokens() == ["<S>", "Here", "'s", "another", "one"]
