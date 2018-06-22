import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.world import ExecutionError
from allennlp.semparse.worlds.atis_world import AtisWorld

class TestAtisWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        # test_filename = self.FIXTURES_ROOT / "data" / "atis" / "sample_data.sql"
        # data = open(test_filename).readlines()

    def test_parse_logical_form(self):
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



        
