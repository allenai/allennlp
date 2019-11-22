from collections import Counter

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import InterleavingDatasetReader
from allennlp.data.iterators import HomogeneousBatchIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.tests.data.dataset_readers.interleaving_dataset_reader_test import PlainTextReader


class TestHomogeneousBatchIterator(AllenNlpTestCase):
    def test_batches(self):
        readers = {"a": PlainTextReader(), "b": PlainTextReader(), "c": PlainTextReader()}

        reader = InterleavingDatasetReader(readers)
        data_dir = self.FIXTURES_ROOT / "data"

        file_path = f"""{{
            "a": "{data_dir / 'babi.txt'}",
            "b": "{data_dir / 'conll2000.txt'}",
            "c": "{data_dir / 'conll2003.txt'}"
        }}"""

        instances = list(reader.read(file_path))
        vocab = Vocabulary.from_instances(instances)

        actual_instance_type_counts = Counter(
            instance.fields["dataset"].metadata for instance in instances
        )

        iterator = HomogeneousBatchIterator(batch_size=3)
        iterator.index_with(vocab)

        observed_instance_type_counts = Counter()

        for batch in iterator(instances, num_epochs=1, shuffle=True):
            # batch should be homogeneous
            instance_types = set(batch["dataset"])
            assert len(instance_types) == 1

            observed_instance_type_counts.update(batch["dataset"])

        assert observed_instance_type_counts == actual_instance_type_counts

    def test_skip_smaller_batches(self):
        readers = {"a": PlainTextReader(), "b": PlainTextReader(), "c": PlainTextReader()}

        reader = InterleavingDatasetReader(readers)
        data_dir = self.FIXTURES_ROOT / "data"

        file_path = f"""{{
            "a": "{data_dir / 'babi.txt'}",
            "b": "{data_dir / 'conll2000.txt'}",
            "c": "{data_dir / 'conll2003.txt'}"
        }}"""

        instances = list(reader.read(file_path))
        vocab = Vocabulary.from_instances(instances)

        iterator = HomogeneousBatchIterator(batch_size=3, skip_smaller_batches=True)
        iterator.index_with(vocab)

        for batch in iterator(instances, num_epochs=1, shuffle=True):
            # every batch should have length 3 (batch size)
            assert len(batch["dataset"]) == 3
