from collections import Counter
import glob
import os
import tarfile
from typing import Tuple

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import (
    SequenceTaggingDatasetReader,
    ShardedDatasetReader,
    DatasetReader,
)
from allennlp.data.instance import Instance


def fingerprint(instance: Instance) -> Tuple[str, ...]:
    """
    Get a hashable representation of a sequence tagging instance
    that can be put in a Counter.
    """
    text_tuple = tuple(t.text for t in instance.fields["tokens"].tokens)  # type: ignore
    labels_tuple = tuple(instance.fields["tags"].labels)  # type: ignore
    return text_tuple + labels_tuple


def test_exception_raised_when_base_reader_implements_sharding():
    class ManuallyShardedBaseReader(DatasetReader):
        def __init__(self, **kwargs):
            super().__init__(manual_distributed_sharding=True, **kwargs)

        def _read(self, file_path: str):
            pass

        def text_to_instance(self, text: str):  # type: ignore
            pass

    with pytest.raises(ValueError, match="should not implement manual distributed sharding"):
        ShardedDatasetReader(ManuallyShardedBaseReader())


class TestShardedDatasetReader(AllenNlpTestCase):
    def setup_method(self) -> None:
        super().setup_method()

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

        # Also create an archive with all of these files to ensure that we can
        # pass the archive directory.
        current_dir = os.getcwd()
        os.chdir(self.TEST_DIR)
        self.archive_filename = self.TEST_DIR / "all_data.tar.gz"
        with tarfile.open(self.archive_filename, "w:gz") as archive:
            for file_path in glob.glob("identical_*.tsv"):
                archive.add(file_path)
        os.chdir(current_dir)

        self.reader = ShardedDatasetReader(base_reader=self.base_reader)

    def read_and_check_instances(self, filepath: str):
        all_instances = []
        for instance in self.reader.read(filepath):
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

    def test_sharded_read_glob(self):
        self.read_and_check_instances(self.identical_files_glob)

    def test_sharded_read_archive(self):
        self.read_and_check_instances(str(self.archive_filename))
