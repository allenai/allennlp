# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.common.testing import AllenNlpTestCase


class TestLanguageModelingDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = LanguageModelingReader(tokens_per_instance=3)

        dataset = reader.read('tests/fixtures/data/language_modeling.txt')
        instances = dataset.instances
        # The last potential instance is left out, which is ok, because we don't have an end token
        # in here, anyway.
        assert len(instances) == 5

        assert instances[0].fields["input_tokens"].tokens == ["This", "is", "a"]
        assert instances[0].fields["output_tokens"].tokens == ["is", "a", "sentence"]

        assert instances[1].fields["input_tokens"].tokens == ["sentence", "for", "language"]
        assert instances[1].fields["output_tokens"].tokens == ["for", "language", "modelling"]

        assert instances[2].fields["input_tokens"].tokens == ["modelling", ".", "Here"]
        assert instances[2].fields["output_tokens"].tokens == [".", "Here", "'s"]

        assert instances[3].fields["input_tokens"].tokens == ["'s", "another", "one"]
        assert instances[3].fields["output_tokens"].tokens == ["another", "one", "for"]

        assert instances[4].fields["input_tokens"].tokens == ["for", "extra", "language"]
        assert instances[4].fields["output_tokens"].tokens == ["extra", "language", "modelling"]
