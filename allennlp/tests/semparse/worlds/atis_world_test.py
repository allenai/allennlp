import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.atis_world import AtisWorld

from allennlp.semparse.contexts.atis_tables import ConversationContext

class TestAtisWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        test_filename = self.FIXTURES_ROOT / "data" / "atis" / "sample.json"
        self.data = open(test_filename).readlines()

    def test_atis_global_actions(self):
        conv_context = ConversationContext()
        valid_actions = conv_context.valid_actions
        # Make sure that the global rules have the expected nonterminals
        assert set(valid_actions.keys()) == {'agg',
                                             'agg_func',
                                             'agg_results',
                                             'and',
                                             'asterisk',
                                             'biexpr',
                                             'binaryop',
                                             'boolean',
                                             'col_ref',
                                             'col_refs',
                                             'condition',
                                             'conditions',
                                             'conj',
                                             'distinct',
                                             'in_clause',
                                             'not',
                                             'number',
                                             'or',
                                             'pos_value',
                                             'query',
                                             'select_results',
                                             'stmt',
                                             'string',
                                             'table_name',
                                             'table_ref',
                                             'table_refs',
                                             'ternaryexpr',
                                             'value',
                                             'where_clause'}

        # Strings and numbers are added locally so they should be initialized to empty sets when
        # we first construct a context
        assert set(valid_actions['string']) == set()
        assert set(valid_actions['number']) == set()

        # Check that valid actions for a nonterminal has the correct production rule
        assert set(valid_actions['query']) == \
                {'(ws "(" ws "SELECT" ws distinct ws select_results ws "FROM" ws table_refs ws where_clause ws ")" ws)',
                 '(ws "SELECT" ws distinct ws select_results ws "FROM" ws table_refs ws where_clause ws)'}

    def test_atis_local_actions(self):
        # Check if the triggers activate correcty
        conv_context = ConversationContext()
        world = AtisWorld(conv_context, "show me the flights from denver at 12 o'clock")
        assert '1200' in world.valid_actions['number']
        assert 'DENVER' in world.valid_actions['string']
        assert 'DEN' in world.valid_actions['string']
        conv_context.valid_actions = world.valid_actions

        world = AtisWorld(conv_context, "show me the delta or united flights in afternoon")
        # Valid local actions from previous utterance should still be valid
        assert '1200' in world.valid_actions['number']
        assert 'DENVER' in world.valid_actions['string']
        assert 'DEN' in world.valid_actions['string']

        # New triggers should be activated
        assert '1800' in world.valid_actions['number']
        assert 'DL' in world.valid_actions['string']
        assert 'UA' in world.valid_actions['string']
        conv_context.valid_actions = world.valid_actions
        world = AtisWorld(conv_context, "i would like one coach reservation for \
                          may ninth from pittsburgh to atlanta leaving \
                          pittsburgh before 10 o'clock in morning 1991 \
                          august twenty sixth")
        assert '26' in world.valid_actions['number'] 
        assert '1000' in world.valid_actions['number']
        assert '1991' in world.valid_actions['number']
        assert '5' in world.valid_actions['number'] 
        assert 'COACH' in world.valid_actions['string']
        assert 'ATLANTA' in world.valid_actions['string']
        assert 'ATL' in world.valid_actions['string']
        assert 'PITTSBURGH' in world.valid_actions['string']
        assert 'PIT' in world.valid_actions['string']

    def test_atis_simple_action_sequence(self):
        conv_context = ConversationContext()
        world = AtisWorld(conv_context, "give me all flights from boston to philadelphia next week arriving after lunch")
        action_sequence = world.get_action_sequence("(SELECT DISTINCT city . city_code , city . city_name FROM city WHERE ( city.city_name = 'BOSTON' ) );")
        assert action_sequence == ['stmt -> [query, ";"]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", "(", conditions, ")"]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> [col_ref, binaryop, value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [string]',
                                   'string -> ["\'BOSTON\'"]',
                                   'binaryop -> ["="]',
                                   'col_ref -> ["city", ".", "city_name"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["city"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref, ",", col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["city", ".", "city_name"]',
                                   'col_ref -> ["city", ".", "city_code"]',
                                   'distinct -> ["DISTINCT"]']

        action_sequence = world.get_action_sequence("( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' ) ) ;")
        assert action_sequence == ['stmt -> [query, ";"]',
                                  'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, where_clause, ")"]',
                                  'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> [col_ref, binaryop, value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [string]',
                                   'string -> ["\'BOSTON\'"]',
                                   'binaryop -> ["="]',
                                   'col_ref -> ["city", ".", "city_name"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["city"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["city", ".", "city_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["airport_service", ".", "city_code"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["airport_service"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["airport_service", ".", "airport_code"]',
                                   'distinct -> [""]']

        action_sequence = world.get_action_sequence("( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' ) AND 1 = 1) ;")
        assert action_sequence == ['stmt -> [query, ";"]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition, conj, conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> [value, binaryop, value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [number]',
                                   'number -> ["1"]',
                                   'binaryop -> ["="]',
                                   'value -> [pos_value]',
                                   'pos_value -> [number]',
                                   'number -> ["1"]',
                                   'conj -> [and]',
                                   'and -> ["AND"]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> [col_ref, binaryop, value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [string]',
                                   'string -> ["\'BOSTON\'"]',
                                   'binaryop -> ["="]',
                                   'col_ref -> ["city", ".", "city_name"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["city"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["city", ".", "city_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["airport_service", ".", "city_code"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["airport_service"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["airport_service", ".", "airport_code"]',
                                   'distinct -> [""]']

        world = AtisWorld(conv_context, "give me all flights from boston to philadelphia next week arriving after lunch")
        action_sequence = world.get_action_sequence(
        """( SELECT DISTINCT flight.flight_id FROM flight WHERE ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )))) ;""")
        assert action_sequence == ['stmt -> [query, ";"]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", "(", conditions, ")"]',
                                   'conditions -> [condition]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> [col_ref, binaryop, value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [string]',
                                   'string -> ["\'BOSTON\'"]',
                                   'binaryop -> ["="]',
                                   'col_ref -> ["city", ".", "city_name"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["city"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["city", ".", "city_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["airport_service", ".", "city_code"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["airport_service"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["airport_service", ".", "airport_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["flight", ".", "from_airport"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["flight"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["flight", ".", "flight_id"]',
                                   'distinct -> ["DISTINCT"]']

    def test_atis_long_action_sequence(self):
        conv_context = ConversationContext()
        world = AtisWorld(conv_context, "what is the earliest flight in morning 1993 june fourth from boston to pittsburgh")
        action_sequence = world.get_action_sequence(
        """( SELECT DISTINCT flight.flight_id FROM flight WHERE ( flight.departure_time = ( SELECT MIN ( flight.departure_time ) FROM flight WHERE ( flight.departure_time BETWEEN 0 AND 1200 AND ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )) AND flight . to_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'PITTSBURGH' )) ) ) ) AND ( flight.departure_time BETWEEN 0 AND 1200 AND ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )) AND flight . to_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'PITTSBURGH' )) ) ) )   ) ;""")
        assert action_sequence == ['stmt -> [query, ";"]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", "(", conditions, ")"]',
                                   'conditions -> [condition, conj, conditions]',
                                   'conditions -> ["(", conditions, ")"]',
                                   'conditions -> [condition, conj, conditions]',
                                   'conditions -> ["(", conditions, ")"]',
                                   'conditions -> [condition, conj, conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> [col_ref, binaryop, value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [string]',
                                   'string -> ["\'PITTSBURGH\'"]',
                                   'binaryop -> ["="]',
                                   'col_ref -> ["city", ".", "city_name"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["city"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["city", ".", "city_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["airport_service", ".", "city_code"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["airport_service"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["airport_service", ".", "airport_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["flight", ".", "to_airport"]',
                                   'conj -> [and]',
                                   'and -> ["AND"]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> [col_ref, binaryop, value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [string]',
                                   'string -> ["\'BOSTON\'"]',
                                   'binaryop -> ["="]',
                                   'col_ref -> ["city", ".", "city_name"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["city"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["city", ".", "city_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["airport_service", ".", "city_code"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["airport_service"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["airport_service", ".", "airport_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["flight", ".", "from_airport"]',
                                   'conj -> [and]',
                                   'and -> ["AND"]',
                                   'condition -> [ternaryexpr]',
                                   'ternaryexpr -> [col_ref, "BETWEEN", value, and value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [number]',
                                   'number -> ["1200"]',
                                   'and -> ["AND"]',
                                   'value -> [pos_value]',
                                   'pos_value -> [number]',
                                   'number -> ["0"]',
                                   'col_ref -> ["flight", ".", "departure_time"]',
                                   'conj -> [and]',
                                   'and -> ["AND"]',
                                   'condition -> [biexpr]',
                                   'biexpr -> [col_ref, binaryop, value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [agg_results]',
                                   'agg_results -> ["(", "SELECT", distinct, agg, "FROM", table_name, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", "(", conditions, ")"]',
                                   'conditions -> [condition, conj, conditions]',
                                   'conditions -> ["(", conditions, ")"]',
                                   'conditions -> [condition, conj, conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> [col_ref, binaryop, value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [string]',
                                   'string -> ["\'PITTSBURGH\'"]',
                                   'binaryop -> ["="]',
                                   'col_ref -> ["city", ".", "city_name"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["city"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["city", ".", "city_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["airport_service", ".", "city_code"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["airport_service"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["airport_service", ".", "airport_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["flight", ".", "to_airport"]',
                                   'conj -> [and]',
                                   'and -> ["AND"]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> [col_ref, binaryop, value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [string]',
                                   'string -> ["\'BOSTON\'"]',
                                   'binaryop -> ["="]',
                                   'col_ref -> ["city", ".", "city_name"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["city"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["city", ".", "city_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["airport_service", ".", "city_code"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["airport_service"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["airport_service", ".", "airport_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["flight", ".", "from_airport"]',
                                   'conj -> [and]',
                                   'and -> ["AND"]',
                                   'condition -> [ternaryexpr]',
                                   'ternaryexpr -> [col_ref, "BETWEEN", value, and value]',
                                   'value -> [pos_value]',
                                   'pos_value -> [number]',
                                   'number -> ["1200"]',
                                   'and -> ["AND"]',
                                   'value -> [pos_value]',
                                   'pos_value -> [number]',
                                   'number -> ["0"]',
                                   'col_ref -> ["flight", ".", "departure_time"]',
                                   'table_name -> ["flight"]',
                                   'agg -> [agg_func, "(", col_ref, ")"]',
                                   'col_ref -> ["flight", ".", "departure_time"]',
                                   'agg_func -> ["MIN"]',
                                   'distinct -> [""]',
                                   'binaryop -> ["="]',
                                   'col_ref -> ["flight", ".", "departure_time"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["flight"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["flight", ".", "flight_id"]',
                                   'distinct -> ["DISTINCT"]']
                        
    def test_atis_from_json(self):
        line = json.loads(self.data[0])
        conv_context = ConversationContext(line['interaction'])
        for interaction_round in conv_context.interaction:
            world = AtisWorld(conv_context, interaction_round['utterance'])
            action_sequence = world.get_action_sequence(interaction_round['sql'])
            conv_context.valid_actions = world.valid_actions
