# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers.wikitables import World
from allennlp.common.testing import AllenNlpTestCase


class TestWorldRepresentation(AllenNlpTestCase):
    def test_world_processes_sempre_forms_correctly(self):
        world = World("tests/fixtures/data/wikitables/sample_table.tsv")
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = world.process_sempre_forms([sempre_form])[0]
        assert str(expression) == "R(C0,C1(cell:usl_a_league))"

    def test_world_returns_correct_actions_with_reverse(self):
        world = World("tests/fixtures/data/wikitables/sample_table.tsv")
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = world.process_sempre_forms([sempre_form])[0]
        actions = world.get_action_sequence(expression)
        target_action_sequence = ['e -> [<r,e>, r]', '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                                  '<<#1,#2>,<#2,#1>> -> R', '<e,r> -> C0', 'r -> [<e,r>, e]',
                                  '<e,r> -> C1', 'e -> cell:usl_a_league']
        assert actions == target_action_sequence

    def test_world_returns_correct_actions_with_two_reverses(self):
        world = World("tests/fixtures/data/wikitables/sample_table.tsv")
        sempre_form = ("(max ((reverse fb:cell.cell.date) ((reverse fb:row.row.year) "
                       "(fb:row.row.league fb:cell.usl_a_league))))")
        expression = world.process_sempre_forms([sempre_form])[0]
        actions = world.get_action_sequence(expression)
        target_action_sequence = ['d -> [<d,d>, d]', '<d,d> -> M0', 'd -> [<e,d>, e]',
                                  '<e,d> -> [<<#1,#2>,<#2,#1>>, <d,e>]', '<<#1,#2>,<#2,#1>> -> R',
                                  '<d,e> -> D', 'e -> [<r,e>, r]', '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                                  '<<#1,#2>,<#2,#1>> -> R', '<e,r> -> C0', 'r -> [<e,r>, e]',
                                  '<e,r> -> C1', 'e -> cell:usl_a_league']
        assert actions == target_action_sequence
