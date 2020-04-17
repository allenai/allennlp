from collections import Counter
from typing import Tuple

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader, ShardedDatasetReader
from allennlp.data.instance import Instance


def fingerprint(instance: Instance) -> Tuple[str, ...]:
    """
    Get a hashable representation of a sequence tagging instance
    that can be put in a Counter.
    """
    text_tuple = tuple(t.text for t in instance.fields["tokens"].tokens)  # type: ignore
    labels_tuple = tuple(instance.fields["tags"].labels)  # type: ignore
    return text_tuple + labels_tuple


class TestShardedDatasetReader(AllenNlpTestCase):
    def setUp(self) -> None:
        super().setUp()

        # use SequenceTaggingDatasetReader as the base reader
        self.base_reader = SequenceTaggingDatasetReader(lazy=True)
        base_file_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"

        # Make 100 copies of the data
        raw_data = open(base_file_path).read()
        for i in range(100):
            file_path = self.TEST_DIR / f"identical_{i}.tsv"
            with open(file_path, "w") as f:
                f.write(raw_data)

        self.identical_files_glob = str(self.TEST_DIR / "identical_*.tsv")

    def test_sharded_read(self):
        reader = ShardedDatasetReader(base_reader=self.base_reader)

        all_instances = []

        for instance in reader.read(self.identical_files_glob):
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
