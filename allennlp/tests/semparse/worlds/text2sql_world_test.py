# pylint: disable=too-many-lines,invalid-name

import sqlite3

from parsimonious import Grammar

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.text2sql_world import Text2SqlWorld
from allennlp.semparse.contexts.sql_context_utils import format_grammar_string
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor


class TestText2SqlWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.schema = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants-schema.csv')
        self.database_path = str(self.FIXTURES_ROOT / "data" / "text2sql" / "restaurants.db")


    def test_world_modifies_unconstrained_grammar_correctly(self):
        world = Text2SqlWorld(self.schema)
        grammar_dictionary = world.base_grammar_dictionary
        assert grammar_dictionary["table_name"] == ['"RESTAURANT"', '"LOCATION"', '"GEOGRAPHIC"']
        assert grammar_dictionary["column_name"] == ['"STREET_NAME"', '"RESTAURANT_ID"', '"REGION"',
                                                     '"RATING"', '"NAME"', '"HOUSE_NUMBER"',
                                                     '"FOOD_TYPE"', '"COUNTY"', '"CITY_NAME"']

    def test_grammar_from_world_can_parse_statements(self):
        world = Text2SqlWorld(self.schema)
        sql = ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', ',',
               'RESTAURANT', 'WHERE', 'LOCATION', '.', 'CITY_NAME', '=',
               "'city_name0'", 'AND', 'RESTAURANT', '.', 'NAME', '=', 'LOCATION',
               '.', 'RESTAURANT_ID', 'AND', 'RESTAURANT', '.', 'NAME', '=', "'name0'", ';']

        grammar = Grammar(format_grammar_string(world.base_grammar_dictionary))
        sql_visitor = SqlVisitor(grammar)
        sql_visitor.parse(" ".join(sql))

    def test_world_adds_values_from_tables(self):
        connection = sqlite3.connect(self.database_path)
        cursor = connection.cursor()
        world = Text2SqlWorld(self.schema, cursor=cursor)
        assert world.base_grammar_dictionary["number"] == ['"229"', '"228"', '"227"', '"226"',
                                                           '"225"', '"5"', '"4"', '"3"', '"2"',
                                                           '"1"', '"833"', '"430"', '"242"',
                                                           '"135"', '"1103"']

        assert world.base_grammar_dictionary["string"] == ['"tommy\'s"', '"rod\'s hickory pit restaurant"',
                                                           '"lyons restaurant"', '"jamerican cuisine"',
                                                           '"denny\'s restaurant"', '"american"', '"vallejo"',
                                                           '"w. el camino real"', '"el camino real"',
                                                           '"e. el camino real"', '"church st"',
                                                           '"broadway"', '"sunnyvale"', '"san francisco"',
                                                           '"san carlos"', '"american canyon"', '"alviso"',
                                                           '"albany"', '"alamo"', '"alameda"', '"unknown"',
                                                           '"santa clara county"', '"contra costa county"',
                                                           '"alameda county"', '"bay area"']
