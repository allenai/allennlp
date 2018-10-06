# pylint: disable=invalid-name
from allennlp.data.dataset_readers.semantic_parsing.grammar_based_text2sql import GrammarBasedText2SqlDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestGrammarBasdText2SqlDatasetReader(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.data_path = str(self.FIXTURES_ROOT / 'data' / 'text2sql'/ '*.json')
        self.schema = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants-schema.csv')
        self.database = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants.db')

        self.reader = GrammarBasedText2SqlDatasetReader(self.schema, self.database)

    def test_reader_can_read_data_with_entity_pre_linking(self):
        instances = self.reader.read(self.data_path)
        instances = list(instances)

        assert len(instances) == 5

        for instance in instances:
            print(instance)