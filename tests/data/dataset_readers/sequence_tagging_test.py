# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.testing.test_case import AllenNlpTestCase


class TestSequenceTaggingDatasetReader(AllenNlpTestCase):

    def setUp(self):
        super(TestSequenceTaggingDatasetReader, self).setUp()
        self.write_sequence_tagging_files()

    def test_read_from_file(self):

        reader = SequenceTaggingDatasetReader(self.TRAIN_FILE)
        dataset = reader.read()

        assert len(dataset.instances) == 4
        fields = dataset.instances[0].fields()
        assert fields["sequence_tokens"].tokens() == ["cats", "are", "animals", "."]
        assert fields["sequence_tags"].tags() == ["N", "V", "N", "N"]
        fields = dataset.instances[1].fields()
        assert fields["sequence_tokens"].tokens() == ["dogs", "are", "animals", "."]
        assert fields["sequence_tags"].tags() == ["N", "V", "N", "N"]
        fields = dataset.instances[2].fields()
        assert fields["sequence_tokens"].tokens() == ["snakes", "are", "animals", "."]
        assert fields["sequence_tags"].tags() == ["N", "V", "N", "N"]
        fields = dataset.instances[3].fields()
        assert fields["sequence_tokens"].tokens() == ["birds", "are", "animals", "."]
        assert fields["sequence_tags"].tags() == ["N", "V", "N", "N"]
