# pylint: disable=no-self-use
from allennlp.data.dataset_readers import NlvrDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestNlvrDatasetReader(AllenNlpTestCase):
    def test_reader_reads(self):
        test_file = "tests/fixtures/data/nlvr/sample_data.jsonl"
        dataset = NlvrDatasetReader().read(test_file)
        assert len(dataset.instances) == 3
        instance = dataset.instances[0]
        assert instance.fields.keys() == {'sentence', 'agenda', 'world', 'actions', 'label'}
