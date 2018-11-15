# pylint: disable=too-many-lines,invalid-name

import sqlite3


from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.text2sql_world import PrelinkedText2SqlWorld, LinkingText2SqlWorld


class TestText2SqlWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.schema = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants-schema.csv')
        self.database_path = str(self.FIXTURES_ROOT / "data" / "text2sql" / "restaurants.db")


    def test_untyped_grammar_has_no_string_or_number_references(self):
        world = PrelinkedText2SqlWorld(self.schema, use_untyped_entities=True)
        grammar_dictionary = world.base_grammar_dictionary

        for key, value in grammar_dictionary.items():
            assert key not in {"number", "string"}
            # We don't check for string directly here because
            # string_set is a valid non-terminal.
            assert all(["number" not in production for production in value])
            assert all(["string)" not in production for production in value])
            assert all(["string " not in production for production in value])
            assert all(["(string " not in production for production in value])

    def test_world_modifies_unconstrained_grammar_tables_correctly(self):
        world = PrelinkedText2SqlWorld(self.schema, use_untyped_entities=True)
        grammar_dictionary = world.base_grammar_dictionary
        assert grammar_dictionary["table_name"] == ['"RESTAURANT"', '"LOCATION"', '"GEOGRAPHIC"']
        assert grammar_dictionary["column_name"] == ['"STREET_NAME"', '"RESTAURANT_ID"', '"REGION"',
                                                     '"RATING"', '"NAME"', '"HOUSE_NUMBER"',
                                                     '"FOOD_TYPE"', '"COUNTY"', '"CITY_NAME"']
    def test_world_modifies_constrained_grammar_tables_correctly(self):
        world = PrelinkedText2SqlWorld(self.schema, use_untyped_entities=False)
        grammar_dictionary = world.base_grammar_dictionary
        assert grammar_dictionary["table_name"] == ['"RESTAURANT"', '"LOCATION"', '"GEOGRAPHIC"']
        assert grammar_dictionary.get("column_name", None) is None
        assert grammar_dictionary["col_ref"] == ['("RESTAURANT" ws "." ws "RESTAURANT_ID")',
                                                 '("RESTAURANT" ws "." ws "RATING")',
                                                 '("RESTAURANT" ws "." ws "NAME")',
                                                 '("RESTAURANT" ws "." ws "FOOD_TYPE")',
                                                 '("RESTAURANT" ws "." ws "CITY_NAME")',
                                                 '("LOCATION" ws "." ws "STREET_NAME")',
                                                 '("LOCATION" ws "." ws "RESTAURANT_ID")',
                                                 '("LOCATION" ws "." ws "HOUSE_NUMBER")',
                                                 '("LOCATION" ws "." ws "CITY_NAME")',
                                                 '("GEOGRAPHIC" ws "." ws "REGION")',
                                                 '("GEOGRAPHIC" ws "." ws "COUNTY")',
                                                 '("GEOGRAPHIC" ws "." ws "CITY_NAME")',
                                                 'table_name']

    def test_world_modifies_grammar_with_global_values_for_dataset(self):
        world = PrelinkedText2SqlWorld(self.schema)
        grammar_dictionary = world.base_grammar_dictionary
        # Should have added 2.5 because it is a global value
        # for the restaurants dataset.
        assert grammar_dictionary["value"] == ['"2.5"', 'parenval', '"YEAR(CURDATE())"',
                                               'boolean', 'function', 'col_ref']

    def test_variable_free_world_cannot_parse_as_statements(self):
        world = PrelinkedText2SqlWorld(self.schema)
        grammar_dictionary = world.base_grammar_dictionary
        for productions in grammar_dictionary.items():
            assert "AS" not in productions

        sql_with_as = ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', 'AS', 'LOCATIONalias0', ',',
                       'RESTAURANT', 'WHERE', 'LOCATION', '.', 'CITY_NAME', '=',
                       "'city_name0'", 'AND', 'RESTAURANT', '.', 'NAME', '=', 'LOCATION',
                       '.', 'RESTAURANT_ID', 'AND', 'RESTAURANT', '.', 'NAME', '=', "'name0'", ';']

        entities = {"city_name0": {"text": "San fran", "type": "location"},
                    "name0": {"text": "Matt Gardinios Pizza", "type": "restaurant"}}
        action_sequence, _, _ = world.get_action_sequence_and_all_actions(sql_with_as, entities)
        assert action_sequence is None

        sql = ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', ',',
               'RESTAURANT', 'WHERE', 'LOCATION', '.', 'CITY_NAME', '=',
               "'city_name0'", 'AND', 'RESTAURANT', '.', 'NAME', '=', 'LOCATION',
               '.', 'RESTAURANT_ID', 'AND', 'RESTAURANT', '.', 'NAME', '=', "'name0'", ';']

        # Without the AS we should still be able to parse it.
        action_sequence, _, _ = world.get_action_sequence_and_all_actions(sql, entities)
        assert action_sequence is not None

    def test_world_identifies_non_global_rules(self):
        world = PrelinkedText2SqlWorld(self.schema)
        assert not world.is_global_rule('value -> ["\'food_type0\'"]')

    def test_untyped_grammar_from_world_can_produce_entities_as_values(self):
        world = PrelinkedText2SqlWorld(self.schema, use_untyped_entities=True)
        sql = ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', ',',
               'RESTAURANT', 'WHERE', 'LOCATION', '.', 'CITY_NAME', '=',
               "'city_name0'", 'AND', 'RESTAURANT', '.', 'NAME', '=', 'LOCATION',
               '.', 'RESTAURANT_ID', 'AND', 'RESTAURANT', '.', 'NAME', '=', "'name0'", ';']

        entities = {"city_name0": {"text": "San fran", "type": "location"},
                    "name0": {"text": "Matt Gardinios Pizza", "type": "restaurant"}}
        action_sequence, actions, _ = world.get_action_sequence_and_all_actions(sql, entities)

        assert 'value -> ["\'city_name0\'"]' in action_sequence
        assert 'value -> ["\'name0\'"]' in action_sequence
        assert 'value -> ["\'city_name0\'"]' in actions
        assert 'value -> ["\'name0\'"]' in actions

    def test_typed_grammar_can_produce_column_constrained_values(self):
        world = PrelinkedText2SqlWorld(self.schema, use_untyped_entities=False)
        sql = ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', ',',
               'RESTAURANT', 'WHERE', 'LOCATION', '.', 'CITY_NAME', '=',
               "'city_name0'", 'AND', 'RESTAURANT', '.', 'NAME', '=', 'LOCATION',
               '.', 'RESTAURANT_ID', 'AND', 'RESTAURANT', '.', 'NAME', '=', "'name0'", ';']

        entities = {"city_name0": {"text": "San fran", "type": "location"},
                    "name0": {"text": "Matt Gardinios Pizza", "type": "restaurant"}}
        action_sequence, actions, _ = world.get_action_sequence_and_all_actions(sql, entities)

        assert 'expr -> ["LOCATION", ".", "CITY_NAME", binaryop, "\'city_name0\'"]' in action_sequence
        assert 'expr -> ["RESTAURANT", ".", "NAME", binaryop, "\'name0\'"]' in action_sequence
        assert 'expr -> ["LOCATION", ".", "CITY_NAME", binaryop, "\'city_name0\'"]' in actions
        assert 'expr -> ["RESTAURANT", ".", "NAME", binaryop, "\'name0\'"]' in actions
