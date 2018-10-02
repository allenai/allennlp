# pylint: disable=invalid-name
from allennlp.semparse.contexts.text2sql_table_context import Text2SqlTableContext
from allennlp.data.dataset_readers.semantic_parsing.grammar_based_text2sql import GrammarBasedText2SqlDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestGrammarBasdText2SqlDatasetReader(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.data_path = str(self.FIXTURES_ROOT / 'data' / 'text2sql'/ '*.json')
        self.schema = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants-schema.csv')

        self.reader = GrammarBasedText2SqlDatasetReader(self.schema)

    def test_reader_can_read_data(self):
        # TODO(Mark): fill in this test.
        _ = self.reader.read(self.data_path)
