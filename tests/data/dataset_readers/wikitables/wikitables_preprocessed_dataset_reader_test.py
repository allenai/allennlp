# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.data.dataset_readers import WikiTablesPreprocessedDatasetReader
from tests.data.dataset_readers.wikitables.wikitables_dataset_reader_test import WikiTablesDatasetReaderTest


class WikiTablesPreprocessedDatasetReaderTest(WikiTablesDatasetReaderTest):
    def test_reader_reads(self):
        # We're going to check that we get the exact same results when reading a pre-processed file
        # as we get when we read the original data (by using the same test code as the original
        # dataset reader).
        reader = WikiTablesPreprocessedDatasetReader()
        dataset = reader.read("tests/fixtures/data/wikitables/sample_data_preprocessed.jsonl")
        self.assert_dataset_correct(dataset)
