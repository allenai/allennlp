# pylint: disable=no-self-use,invalid-name
from typing import Tuple
from collections import Counter

from allennlp.data.dataset_readers import MultiprocessDatasetReader, SequenceTaggingDatasetReader
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.testing import AllenNlpTestCase

def fingerprint(instance: Instance) -> Tuple[str, ...]:
    """
    Get a hashable representation of a sequence tagging instance
    that can be put in a Counter.
    """
    text_tuple = tuple(t.text for t in instance.fields["tokens"].tokens)  # type: ignore
    labels_tuple = tuple(instance.fields["tags"].labels)                  # type: ignore
    return text_tuple + labels_tuple


class TestMultiprocessDatasetReader(AllenNlpTestCase):
    def setUp(self) -> None:
        super().setUp()

        # use SequenceTaggingDatasetReader as the base reader
        self.base_reader = SequenceTaggingDatasetReader(lazy=True)
        base_file_path = AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv'


        # Make 100 copies of the data
        raw_data = open(base_file_path).read()
        for i in range(100):
            file_path = self.TEST_DIR / f'sequence_tagging_{i}.tsv'
            with open(file_path, 'w') as f:
                f.write(raw_data)

        self.glob = str(self.TEST_DIR / 'sequence_tagging_*.tsv')

        # For some of the tests we need a vocab, we'll just use the base_reader for that.
        self.vocab = Vocabulary.from_instances(self.base_reader.read(str(base_file_path)))

    def test_multiprocess_read(self):
        reader = MultiprocessDatasetReader(base_reader=self.base_reader, num_workers=4)

        all_instances = []

        for instance in reader.read(self.glob):
            all_instances.append(instance)

        # 100 files * 4 sentences / file
        assert len(all_instances) == 100 * 4

        counts = Counter(fingerprint(instance) for instance in all_instances)

        # should have the exact same data 100 times
        assert len(counts) == 4
        assert counts[("cats", "are", "animals", ".", "N", "V", "N", "N")] == 100
        assert counts[("dogs", "are", "animals", ".", "N", "V", "N", "N")] == 100
        assert counts[("snakes", "are", "animals", ".", "N", "V", "N", "N")] == 100
        assert counts[("birds", "are", "animals", ".", "N", "V", "N", "N")] == 100

    def test_multiple_epochs(self):
        reader = MultiprocessDatasetReader(base_reader=self.base_reader,
                                           num_workers=2,
                                           epochs_per_read=3)

        all_instances = []

        for instance in reader.read(self.glob):
            all_instances.append(instance)

        # 100 files * 4 sentences per file * 3 epochs
        assert len(all_instances) == 100 * 4 * 3

        counts = Counter(fingerprint(instance) for instance in all_instances)

        # should have the exact same data 100 * 3 times
        assert len(counts) == 4
        assert counts[("cats", "are", "animals", ".", "N", "V", "N", "N")] == 300
        assert counts[("dogs", "are", "animals", ".", "N", "V", "N", "N")] == 300
        assert counts[("snakes", "are", "animals", ".", "N", "V", "N", "N")] == 300
        assert counts[("birds", "are", "animals", ".", "N", "V", "N", "N")] == 300

    def test_with_iterator(self):
        reader = MultiprocessDatasetReader(base_reader=self.base_reader, num_workers=2)
        instances = reader.read(self.glob)

        iterator = BasicIterator(batch_size=32)
        iterator.index_with(self.vocab)

        batches = [batch for batch in iterator(instances, num_epochs=1)]

        # 400 instances / batch_size 32 = 12 full batches + 1 batch of 16
        sizes = sorted([len(batch['tags']) for batch in batches])
        assert sizes == [16] + 12 * [32]
