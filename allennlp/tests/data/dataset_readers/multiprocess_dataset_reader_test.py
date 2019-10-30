from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params
from allennlp.data.dataset_readers import (
    MultiprocessDatasetReader,
    SequenceTaggingDatasetReader,
    DatasetReader,
)


class TestMultiprocessDatasetReader(AllenNlpTestCase):
    def test_construction_returns_base_reader(self):

        reader = MultiprocessDatasetReader(SequenceTaggingDatasetReader(), 10)
        assert isinstance(reader, SequenceTaggingDatasetReader)

    def test_construction_returns_base_reader_from_params(self):

        params = Params(
            {"type": "multiprocess", "base_reader": {"type": "sequence_tagging"}, "num_workers": 2}
        )

        reader = DatasetReader.from_params(params)
        assert isinstance(reader, SequenceTaggingDatasetReader)
