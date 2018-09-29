# pylint: disable=invalid-name
from allennlp.semparse.contexts.text2sql_table_context import WeaklyConstrainedText2SqlTableContext
from allennlp.common.testing import AllenNlpTestCase


class TestText2sqlTableContext(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.schema = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants-schema.csv')

    def test_context_modifies_unconstrained_grammar_correctly(self):
        context = WeaklyConstrainedText2SqlTableContext(self.schema)
        grammar_dictionary = context.get_grammar_dictionary()
        assert grammar_dictionary["table_name"] == ['"RESTAURANT"', '"LOCATION"', '"GEOGRAPHIC"']
        assert grammar_dictionary["column_name"] == ['"STREET_NAME"', '"RESTAURANT_ID"', '"REGION"',
                                                     '"RATING"', '"NAME"', '"HOUSE_NUMBER"',
                                                     '"FOOD_TYPE"', '"COUNTY"', '"CITY_NAME"']
