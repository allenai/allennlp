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
        conv_context = ConversationContext(None)
        valid_actions = conv_context.valid_actions

        assert set(valid_actions.keys()) == {'stmt',
                                             'query',
                                             'select_results',
                                             'agg',
                                             'agg_func',
                                             'col_refs',
                                             'table_refs',
                                             'where_clause',
                                             'condition_paren',
                                             'condition_paren2',
                                             'condition_paren3',
                                             'condition',
                                             'in_clause',
                                             'biexpr',
                                             'binaryop',
                                             'ternaryexpr',
                                             'value',
                                             'pos_value',
                                             'agg_results',
                                             'boolean',
                                             'lparen',
                                             'rparen',
                                             'conj',
                                             'and',
                                             'or',
                                             'not',
                                             'asterisk',
                                             'col_ref',
                                             'table_ref',
                                             'table_name',
                                             'number',
                                             'string'}

        assert set(valid_actions['string']) == set()
        
        assert set(valid_actions['query']) == \
                set(['ws lparen? ws "SELECT" ws "DISTINCT"? ws select_results ws "FROM" ws table_refs ws where_clause rparen? ws'])
        

    def test_atis_local_actions(self):
        conv_context = ConversationContext(None)
        world = AtisWorld(conv_context, "show me the flights from denver at 12 o'clock")
        assert '1200' in world.valid_actions['number']
        assert 'DENVER' in world.valid_actions['string']
        conv_context.valid_actions = world.valid_actions

        world = AtisWorld(conv_context, "show me the delta or united flights in afternoon")
        assert '1800' in world.valid_actions['number'] 
        assert 'DL' in world.valid_actions['string']
        assert 'UA' in world.valid_actions['string']
        conv_context.valid_actions = world.valid_actions
 
        world = AtisWorld(conv_context, "i would like one coach reservation for \
                          may ninth from pittsburgh to atlanta leaving \
                          pittsburgh before 10 o'clock in morning 1991 \
                          august twenty sixth")
        assert '26' in world.valid_actions['number'] 
        assert 'COACH' in world.valid_actions['string']


    def test_simple_sequence(self):
        conv_context = ConversationContext(None)
        world = AtisWorld(conv_context, "give me all flights from boston to philadelphia next week arriving after lunch")
        action_sequence = world.get_action_sequence("(SELECT DISTINCT city . city_code FROM city WHERE ( city.city_name = 'BOSTON' ) );")
        print(action_sequence)

        action_sequence = world.get_action_sequence("( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' ) ) ;")
        print(action_sequence)

        conv_context = ConversationContext(None)
        world = AtisWorld(conv_context, "give me all flights from boston to philadelphia next week arriving after lunch")

        action_sequence = world.get_action_sequence(
        """( SELECT DISTINCT flight.flight_id FROM flight WHERE ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )) AND ( flight . to_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'PHILADELPHIA' )) AND flight.arrival_time > 1400 ) ) ) ;""")
        print(action_sequence)

    def test_conjunctions(self):
        conv_context = ConversationContext(None)
        world = AtisWorld(conv_context, "give me all flights from boston to philadelphia next week arriving after lunch")

        action_sequence = world.get_action_sequence(
        """flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )) AND ( flight . to_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'PHILADELPHIA' )) AND flight.arrival_time > 1400 ) """)
        print(action_sequence)


       
    def test_atis_action_sequence(self):

        conv_context = ConversationContext(None)
        world = AtisWorld(conv_context, "give me all flights from boston to philadelphia next week arriving after lunch")
        action_sequence = world.get_action_sequence("(SELECT DISTINCT city . city_code FROM city WHERE city.city_name = 'BOSTON');")
        print(action_sequence)
        action_sequence = world.get_action_sequence("( SELECT DISTINCT flight.flight_id FROM flight WHERE ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' ) ) ) );")
        print(action_sequence)
        
        conv_context = ConversationContext(None)
        world = AtisWorld(conv_context, "give me all flights from boston to philadelphia next week arriving after lunch")

        action_sequence = world.get_action_sequence(
        """( SELECT DISTINCT flight.flight_id FROM flight WHERE ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )) AND ( flight . to_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'PHILADELPHIA' )) AND flight.arrival_time > 1400 ) ) ) ;""")

        world = AtisWorld(conv_context, "what is the earliest flight in morning 1993 june fourth from boston to pittsburgh")
        action_sequence = world.get_action_sequence(
        """( SELECT DISTINCT flight.flight_id FROM flight WHERE ( flight.departure_time = ( SELECT MIN ( flight.departure_time ) FROM flight WHERE ( flight.departure_time BETWEEN 0 AND 1200 AND ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )) AND flight . to_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'PITTSBURGH' )) ) ) ) AND ( flight.departure_time BETWEEN 0 AND 1200 AND ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )) AND flight . to_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'PITTSBURGH' )) ) ) )   ) ;""")
    
    def test_atis_from_json(self):
        line = json.loads(self.data[0])
        conv_context = ConversationContext(line['interaction'])

        for interaction_round in conv_context.interaction:
            world = AtisWorld(conv_context, interaction_round['utterance'])
            action_sequence = world.get_action_sequence(interaction_round['sql'])
            conv_context.valid_actions = world.valid_actions
            print(action_sequence)

    def test_atis_parse_coverage(self):
        print(self.data)
        num_queries = 0
        num_parsed = 0

        for idx, line in enumerate(self.data):
            jline = json.loads(line)
            conv_context = ConversationContext(jline['interaction'])

            for interaction_round in conv_context.interaction:
                world = AtisWorld(conv_context, interaction_round['utterance'])

                try:
                    num_queries += 1
                    action_sequence = world.get_action_sequence(interaction_round['sql'])
                    num_parsed += 1
                except:
                    print(line)
                    print("Failed to parse, line {}".format(idx))

                conv_context.valid_actions = world.valid_actions

        print("Parsed {} out of {}, coverage: {}".format(num_parsed, num_queries, num_parsed/num_queries))

