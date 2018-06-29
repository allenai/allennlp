import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.world import ExecutionError
from allennlp.semparse.worlds.atis_world import AtisWorld

class TestAtisWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        # test_filename = self.FIXTURES_ROOT / "data" / "atis" / "sample_data.sql"
        # data = open(test_filename).readlines()

    def test_atis_valid_actions(self):
        world = AtisWorld("show me the flights from baltimore to denver") 
        print(world.get_valid_actions())

        world = AtisWorld("show me the delta or united flights")
        print(world.get_valid_actions())

        world = AtisWorld("i would like one coach reservation for may ninth from pittsburgh to atlanta leaving pittsburgh before 10 o'clock in morning 1991 august twenty sixth")
        print(world.get_valid_actions())

    
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







        

        

        






        
