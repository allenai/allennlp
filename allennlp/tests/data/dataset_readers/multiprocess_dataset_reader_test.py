from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import MultiprocessDatasetReader, SequenceTaggingDatasetReader


class TestMultiprocessDatasetReader(AllenNlpTestCase):
    def test_construction_returns_base_reader(self):

        reader = MultiprocessDatasetReader(SequenceTaggingDatasetReader(), 10)
        assert isinstance(reader, SequenceTaggingDatasetReader)
