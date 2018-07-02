import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.world import ExecutionError
from allennlp.semparse.worlds.atis_world import AtisWorld


from allennlp.semparse.contexts.atis_tables import ConversationContext  

class TestAtisWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        test_filename = self.FIXTURES_ROOT / "data" / "atis" / "sample.json"
        self.data = open(test_filename).readlines()

    def test_atis_foo(self):
        conv_context = ConversationContext(None)
        world = AtisWorld(conv_context, "show me the flights from denver at 12 o'clock") 

    def test_atis_parse_coverage(self):
        num_queries = 0
        num_parsed = 0

        for line in self.data:
            jline = json.loads(line)
            conv_context = ConversationContext(jline['interaction'])

            for interaction_round in conv_context.interaction:
                print(interaction_round)

                world = AtisWorld(conv_context, interaction_round['utterance']) 

                try:
                    num_queries += 1
                    action_sequence = world.get_action_sequence(interaction_round['sql'])
                    print(action_sequence)
                    num_parsed += 1
                except:
                    print("Failed to parse")
                    pass

                conv_context.valid_actions = world.valid_actions

        print("Parsed {} out of {}, coverage: {}".format(num_parsed, num_queries, num_parsed/num_queries))


    def test_atis_with_context(self):
        line = json.loads(self.data[12])
        conv_context = ConversationContext(line['interaction'])

        for interaction_round in conv_context.interaction:
            print(interaction_round)

            world = AtisWorld(conv_context, interaction_round['utterance']) 
            action_sequence = world.get_action_sequence(interaction_round['sql'])
            conv_context.valid_actions = world.valid_actions
            print(action_sequence)


    def test_atis_valid_actions(self):
        world = AtisWorld("show me the flights from baltimore to denver") 
        print(world.get_valid_actions())

        world = AtisWorld("show me the delta or united flights")
        print(world.get_valid_actions())

        world = AtisWorld("i would like one coach reservation for may ninth from pittsburgh to atlanta leaving pittsburgh before 10 o'clock in morning 1991 august twenty sixth")
        print(world.get_valid_actions())

    def test_atis_table_column_names(self):
        world = AtisWorld("")
        action_sequence = world.get_action_sequence("( SELECT DISTINCT flight.stops FROM flight WHERE ( flight.airline_code = 'DL') )  ;")
        print(action_sequence)
    
    def test_atis_table_valid_actions(self):
        world = AtisWorld("give me all flights from boston to philadelphia next week arriving after lunch")
        print(world.get_valid_actions())

    def test_atis_table_action_seq(self):
        world = AtisWorld("give me all flights from boston to philadelphia next week arriving after lunch")
        action_sequence = world.get_action_sequence(
        """( SELECT DISTINCT flight.flight_id FROM flight WHERE ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )) AND ( flight . to_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'PHILADELPHIA' )) AND flight.arrival_time > 1400 ) ) ) ;""")
        print(action_sequence)


        world = AtisWorld("what is the earliest flight in morning 1993 june fourth from boston to pittsburgh")
        action_sequence = world.get_action_sequence(
        """( SELECT DISTINCT flight.flight_id FROM flight WHERE ( flight.departure_time = ( SELECT MIN ( flight.departure_time ) FROM flight WHERE ( flight.departure_time BETWEEN 0 AND 1200 AND ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )) AND flight . to_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service
        . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'PITTSBURGH' )) ) ) ) AND ( flight.departure_time BETWEEN 0 AND 1200 AND ( flight . from_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city . city_code FROM city WHERE city.city_name = 'BOSTON' )) AND flight . to_airport IN ( SELECT airport_service . airport_code FROM airport_service WHERE airport_service . city_code IN ( SELECT city .
        city_code FROM city WHERE city.city_name = 'PITTSBURGH' )) ) ) )   ) ;""")
        print(action_sequence)
    
    
    def test_atis_get_context_str(self):
        world = AtisWorld("give me all flights from boston to philadelphia next week arriving after lunch")
        context_str = world.get_grammar_str_with_context()
        print(context_str)

    def test_atis_parse_strings(self):
        # world = self.worlds[0]
        world = AtisWorld() 
        parsed = world.parse_logical_form("")
        print(world.get_action_sequence(parsed))
 
    
    def test_atis_parse_strings(self):
        # world = self.worlds[0]
        world = AtisWorld() 
        parsed = world.parse_logical_form("(>= string:'NEW_YORK_CITY' num:1234)")
        print(world.get_action_sequence(parsed))
     
    def test_atis_parse_logical_form(self):
        # world = self.worlds[0]
        world = AtisWorld() 
        
        parsed = world.parse_logical_form("(>= num:1234 num:1234)")
        print(world.get_action_sequence(parsed))
        parsed = world.parse_logical_form("(WHERE (>= num:1234 num:1234))")
        print(world.get_action_sequence(parsed))
        parsed = world.parse_logical_form("(FROM string:flight)")
        print(world.get_action_sequence(parsed))
        parsed = world.parse_logical_form("(SELECT string:flight_arrival_time (FROM string:flight) (WHERE (>= num:1234 num:1234)))")
        print(world.get_action_sequence(parsed))
        
        
        parsed = world.parse_logical_form("(SELECT_DISTINCT string:flight_arrival_time (FROM string:flight) (WHERE (>= num:1234 num:1234)))")
        print(world.get_action_sequence(parsed))

    def test_atis_parse_conjunctions(self):
        world = AtisWorld()
        parsed = world.parse_logical_form("(AND (>= num:1234 num:1234) (>= num:1234 num:1234))")
        print(world.get_action_sequence(parsed))

        parsed = world.parse_logical_form("(AND (AND (>= num:1234 num:1234) (>= num:1234 num:1234)) (>= num:1234 num:1234))")
        print(world.get_action_sequence(parsed))


    def test_atis_parse_nested(self):
        world = AtisWorld() 
        parsed = world.parse_logical_form("(IN string:flight_arrival_time (SELECT string:flight_arrival_time (FROM string:flight) (WHERE (>= num:1234 num:1234))))") 
        print(world.get_action_sequence(parsed))

        parsed = world.parse_logical_form("(SELECT string:flight_arrival_time (FROM string:flight) (WHERE (IN string:flight_arrival_time (SELECT string:flight_arrival_time (FROM string:flight) (WHERE (>= num:1234 num:1234))))))") 
        print(world.get_action_sequence(parsed))

    def test_atis_parse_distinct(self):
        world = AtisWorld() 
        parsed = world.parse_logical_form("(SELECT_DISTINCT string:flight_flight_id (FROM string:flight) (WHERE   (IN string:flight_from_airport (SELECT string:airport_service_airport_code (FROM string:airport_service) (WHERE   (IN string:airport_service_city_code (SELECT string:city_city_code (FROM string:city) (WHERE   (= string:'KANSAS_CITY' string:city_city_name)))))))))")
        print(world.get_action_sequence(parsed))

