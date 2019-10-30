from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader, MultiprocessDatasetReader
from allennlp.data.iterators import MultiprocessIterator, TransformIterator
from allennlp.data.iterators.basic_iterator import BasicIteratorStub
from allennlp.data import transforms
from allennlp.data.vocabulary import Vocabulary
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


class TestMultiprocessIterator(IteratorTest):
    def setUp(self):
        super().setUp()

        self.base_reader = SequenceTaggingDatasetReader(lazy=True)
        base_file_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"

        # Make 100 copies of the data
        raw_data = open(base_file_path).read()
        for i in range(100):
            file_path = self.TEST_DIR / f"sequence_tagging_{i}.tsv"
            with open(file_path, "w") as f:
                f.write(raw_data)

        self.glob = str(self.TEST_DIR / "sequence_tagging_*.tsv")

        # For some of the tests we need a vocab, we'll just use the base_reader for that.
        self.vocab = Vocabulary.from_instances(self.base_reader.read(str(base_file_path)))

    def test_construction_returns_modified_base_iterator(self):
        iterator = MultiprocessIterator(BasicIteratorStub(batch_size=32), num_workers=3)
        assert isinstance(iterator, TransformIterator)
        assert isinstance(iterator.transforms[-1], transforms.Fork)
        assert iterator._num_workers == 3
