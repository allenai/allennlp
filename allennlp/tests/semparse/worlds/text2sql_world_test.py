# pylint: disable=too-many-lines,invalid-name

import sqlite3

from parsimonious import Grammar, ParseError

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.text2sql_world import Text2SqlWorld
from allennlp.semparse.contexts.sql_context_utils import format_grammar_string
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor


class TestText2SqlWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.schema = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants-schema.csv')
        self.database_path = str(self.FIXTURES_ROOT / "data" / "text2sql" / "restaurants.db")

    def test_untyped_grammar_has_no_string_or_number_references(self):
        world = Text2SqlWorld(self.schema, use_untyped_entities=True)
        grammar_dictionary = world.base_grammar_dictionary

        for key, value in grammar_dictionary.items():
            assert key not in {"number", "string"}
            # We don't check for string directly here because
            # string_set is a valid non-terminal.
            assert all(["number" not in production for production in value])
            assert all(["string)" not in production for production in value])
            assert all(["string " not in production for production in value])
            assert all(["(string " not in production for production in value])

    def test_world_modifies_unconstrained_grammar_correctly(self):
        world = Text2SqlWorld(self.schema)
        grammar_dictionary = world.base_grammar_dictionary
        assert grammar_dictionary["table_name"] == ['"RESTAURANT"', '"LOCATION"', '"GEOGRAPHIC"']
        assert grammar_dictionary["column_name"] == ['"STREET_NAME"', '"RESTAURANT_ID"', '"REGION"',
                                                     '"RATING"', '"NAME"', '"HOUSE_NUMBER"',
                                                     '"FOOD_TYPE"', '"COUNTY"', '"CITY_NAME"']

    def test_world_modifies_grammar_with_global_values_for_dataset(self):
        world = Text2SqlWorld(self.schema)
        grammar_dictionary = world.base_grammar_dictionary
        # Should have added 2.5 because it is a global value
        # for the restaurants dataset.
        assert grammar_dictionary["value"] == ['"2.5"', 'parenval', '"YEAR(CURDATE())"',
                                               'number', 'boolean', 'function', 'col_ref', 'string']

    def test_variable_free_world_cannot_parse_as_statements(self):
        world = Text2SqlWorld(self.schema)
        grammar_dictionary = world.base_grammar_dictionary
        for productions in grammar_dictionary.items():
            assert "AS" not in productions

        sql_with_as = ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', 'AS', 'LOCATIONalias0', ',',
                       'RESTAURANT', 'WHERE', 'LOCATION', '.', 'CITY_NAME', '=',
                       "'city_name0'", 'AND', 'RESTAURANT', '.', 'NAME', '=', 'LOCATION',
                       '.', 'RESTAURANT_ID', 'AND', 'RESTAURANT', '.', 'NAME', '=', "'name0'", ';']

        grammar = Grammar(format_grammar_string(world.base_grammar_dictionary))
        sql_visitor = SqlVisitor(grammar)

        with self.assertRaises(ParseError):
            sql_visitor.parse(" ".join(sql_with_as))

        sql = ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', ',',
               'RESTAURANT', 'WHERE', 'LOCATION', '.', 'CITY_NAME', '=',
               "'city_name0'", 'AND', 'RESTAURANT', '.', 'NAME', '=', 'LOCATION',
               '.', 'RESTAURANT_ID', 'AND', 'RESTAURANT', '.', 'NAME', '=', "'name0'", ';']

        # Without the AS we should still be able to parse it.
        sql_visitor = SqlVisitor(grammar)
        sql_visitor.parse(" ".join(sql))

    def test_grammar_from_world_can_parse_statements(self):
        world = Text2SqlWorld(self.schema)
        sql = ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', ',',
               'RESTAURANT', 'WHERE', 'LOCATION', '.', 'CITY_NAME', '=',
               "'city_name0'", 'AND', 'RESTAURANT', '.', 'NAME', '=', 'LOCATION',
               '.', 'RESTAURANT_ID', 'AND', 'RESTAURANT', '.', 'NAME', '=', "'name0'", ';']

        grammar = Grammar(format_grammar_string(world.base_grammar_dictionary))
        sql_visitor = SqlVisitor(grammar)
        sql_visitor.parse(" ".join(sql))

    def test_world_identifies_non_global_rules(self):
        world = Text2SqlWorld(self.schema)
        assert not world.is_global_rule('value -> ["\'food_type0\'"]')

    def test_grammar_from_world_can_produce_entities_as_values(self):
        world = Text2SqlWorld(self.schema)
        sql = ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', ',',
               'RESTAURANT', 'WHERE', 'LOCATION', '.', 'CITY_NAME', '=',
               "'city_name0'", 'AND', 'RESTAURANT', '.', 'NAME', '=', 'LOCATION',
               '.', 'RESTAURANT_ID', 'AND', 'RESTAURANT', '.', 'NAME', '=', "'name0'", ';']

        entities = {"city_name0": {"text": "San fran", "type": "location"},
                    "name0": {"text": "Matt Gardinios Pizza", "type": "restaurant"}}
        action_sequence, actions = world.get_action_sequence_and_all_actions(sql, entities)
        assert 'string -> ["\'city_name0\'"]' in action_sequence
        assert 'string -> ["\'name0\'"]' in action_sequence
        assert 'string -> ["\'city_name0\'"]' in actions
        assert 'string -> ["\'name0\'"]' in actions

    def test_world_adds_values_from_tables(self):
        connection = sqlite3.connect(self.database_path)
        cursor = connection.cursor()
        world = Text2SqlWorld(self.schema, cursor=cursor, use_prelinked_entities=False)
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
