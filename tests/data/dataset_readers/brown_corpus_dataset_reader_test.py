# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestBrownCorpusDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = SequenceTaggingDatasetReader(word_tag_delimiter='/', word_word_delimiter=' ')
        dataset = reader.read('tests/fixtures/data/brown_corpus.txt')

        assert len(dataset.instances) == 4
        fields = dataset.instances[0].fields()
        assert fields["tokens"].tokens() == ["cats", "are", "animals", "."]
        assert fields["tags"].tags() == ["N", "V", "N", "N"]
        fields = dataset.instances[1].fields()
        assert fields["tokens"].tokens() == ["dogs", "are", "animals", "."]
        assert fields["tags"].tags() == ["N", "V", "N", "N"]
        fields = dataset.instances[2].fields()
        assert fields["tokens"].tokens() == ["snakes", "are", "animals", "."]
        assert fields["tags"].tags() == ["N", "V", "N", "N"]
        fields = dataset.instances[3].fields()
        assert fields["tokens"].tokens() == ["birds", "are", "animals", "."]
        assert fields["tags"].tags() == ["N", "V", "N", "N"]
