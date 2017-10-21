# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers.wikitables import World
from allennlp.common.testing import AllenNlpTestCase


class TestWorldRepresentation(AllenNlpTestCase):
    def test_world_processes_sempre_forms_correctly(self):
        world = World("tests/fixtures/data/wikitables/sample_table.tsv")
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = world.process_sempre_forms([sempre_form])[0]
        assert str(expression) == "R(C0,C1(cell:usl_a_league))"

    def test_world_returns_correct_action_sequences(self):
        world = World("tests/fixtures/data/wikitables/sample_table.tsv")
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = world.process_sempre_forms([sempre_form])[0]
        actions = world.get_action_sequence(expression)
        target_action_sequence = ['e -> [<r,e>, r]', '<r,e> -> [<<e,r>,<r,e>>, <e,r>]', '<<e,r>,<r,e>> -> R',
                                  '<e,r> -> C0', 'r -> [<e,r>, e]', '<e,r> -> C1', 'e -> cell:usl_a_league']
        assert actions == target_action_sequence
