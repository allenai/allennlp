from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader, MultiprocessDatasetReader
from allennlp.data.iterators import BasicIterator, MultiprocessIterator
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

    def test_yield_one_epoch_iterates_over_the_data_once(self):
        for test_instances in (self.instances, self.lazy_instances):
            base_iterator = BasicIterator(batch_size=2, max_instances_in_memory=1024)
            iterator = MultiprocessIterator(base_iterator, num_workers=4)
            iterator.index_with(self.vocab)
            batches = list(iterator(test_instances, num_epochs=1))
            # We just want to get the single-token array for the text field in the instance.
            instances = [
                tuple(instance.detach().cpu().numpy())
                for batch in batches
                for instance in batch["text"]["tokens"]["tokens"]
            ]
            assert len(instances) == 5

    def test_multiprocess_iterate_partial_does_not_hang(self):
        for test_instances in (self.instances, self.lazy_instances):
            base_iterator = BasicIterator(batch_size=2, max_instances_in_memory=1024)
            iterator = MultiprocessIterator(base_iterator, num_workers=4)
            iterator.index_with(self.vocab)
            generator = iterator(test_instances, num_epochs=1)
            # We only iterate through 3 of the 5 instances causing the
            # processes generating the tensors to remain active.
            for _ in range(3):
                next(generator)
            # The real test here is that we exit normally and don't hang due to
            # the still active processes.

    def test_multiprocess_reader_with_multiprocess_iterator(self):
        # use SequenceTaggingDatasetReader as the base reader
        reader = MultiprocessDatasetReader(base_reader=self.base_reader, num_workers=2)
        base_iterator = BasicIterator(batch_size=32, max_instances_in_memory=1024)

        iterator = MultiprocessIterator(base_iterator, num_workers=2)
        iterator.index_with(self.vocab)

        instances = reader.read(self.glob)

        tensor_dicts = iterator(instances, num_epochs=1)
        sizes = [len(tensor_dict["tags"]) for tensor_dict in tensor_dicts]
        assert sum(sizes) == 400
