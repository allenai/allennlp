# pylint: disable=invalid-name

from allennlp.semparse.contexts.text2sql_table_context import Text2SqlTableContext
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor
from allennlp.common.testing import AllenNlpTestCase


class TestText2sqlTableContext(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.schema = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants-schema.csv')

    def test_context_modifies_unconstrained_grammar_correctly(self):
        context = Text2SqlTableContext(self.schema)
        grammar_dictionary = context.get_grammar_dictionary()
        assert grammar_dictionary["table_name"] == ['"RESTAURANT"', '"LOCATION"', '"GEOGRAPHIC"']
        assert grammar_dictionary["column_name"] == ['"STREET_NAME"', '"RESTAURANT_ID"', '"REGION"',
                                                     '"RATING"', '"NAME"', '"HOUSE_NUMBER"',
                                                     '"FOOD_TYPE"', '"COUNTY"', '"CITY_NAME"']

    def test_grammar_from_context_can_parse_statements(self):
        context = Text2SqlTableContext(self.schema)
        sql = ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', ',',
               'RESTAURANT', 'WHERE', 'LOCATION', '.', 'CITY_NAME', '=',
               "'city_name0'", 'AND', 'RESTAURANT', '.', 'NAME', '=', 'LOCATION',
               '.', 'RESTAURANT_ID', 'AND', 'RESTAURANT', '.', 'NAME', '=', "'name0'", ';']

        sql_visitor = SqlVisitor(context.grammar)
        sql_visitor.parse(" ".join(sql))
