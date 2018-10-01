# pylint: disable=invalid-name
from allennlp.semparse.contexts.text2sql_table_context import WeaklyConstrainedText2SqlTableContext
from allennlp.data.dataset_readers.semantic_parsing.grammar_based_text2sql import GrammarBasedText2SqlDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestGrammarBasdText2SqlDatasetReader(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.data_path = str(self.FIXTURES_ROOT / 'data' / 'text2sql'/ '*.json')
        self.schema = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants-schema.csv')

        context = WeaklyConstrainedText2SqlTableContext(self.schema)
        self.reader = GrammarBasedText2SqlDatasetReader(context)

    def test_reader_can_read_data(self):

        instances = self.reader.read(self.data_path)