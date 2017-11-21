# pylint: disable=no-self-use,invalid-name
from allennlp.data.semparse.knowledge_graphs.table_knowledge_graph import TableKnowledgeGraph
from allennlp.data.semparse.worlds import WikiTablesWorld
from allennlp.common.testing import AllenNlpTestCase


class TestWikiTablesWorldRepresentation(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        table_kg = TableKnowledgeGraph.read_from_file("tests/fixtures/data/wikitables/sample_table.tsv")
        self.world = WikiTablesWorld(table_kg)

    def test_world_processes_sempre_forms_correctly(self):
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world.process_sempre_forms([sempre_form])[0]
        assert str(expression) == "R(C0,C1(cell:usl_a_league))"

    def test_world_returns_correct_actions_with_reverse(self):
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world.process_sempre_forms([sempre_form])[0]
        actions = self.world.get_action_sequence(expression)
        target_action_sequence = ['e', 'e -> [<r,e>, r]', '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                                  '<<#1,#2>,<#2,#1>> -> R', '<e,r> -> C0', 'r -> [<e,r>, e]',
                                  '<e,r> -> C1', 'e -> cell:usl_a_league']
        assert actions == target_action_sequence

    def test_world_returns_correct_actions_with_two_reverses(self):
        sempre_form = ("(max ((reverse fb:cell.cell.date) ((reverse fb:row.row.year) "
                       "(fb:row.row.league fb:cell.usl_a_league))))")
        expression = self.world.process_sempre_forms([sempre_form])[0]
        actions = self.world.get_action_sequence(expression)
        target_action_sequence = ['d', 'd -> [<d,d>, d]', '<d,d> -> M0', 'd -> [<e,d>, e]',
                                  '<e,d> -> [<<#1,#2>,<#2,#1>>, <d,e>]', '<<#1,#2>,<#2,#1>> -> R',
                                  '<d,e> -> D1', 'e -> [<r,e>, r]', '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                                  '<<#1,#2>,<#2,#1>> -> R', '<e,r> -> C0', 'r -> [<e,r>, e]',
                                  '<e,r> -> C1', 'e -> cell:usl_a_league']
        assert actions == target_action_sequence

    def test_large_scale_processing(self):
        # A sample of 500 logical forms taken randomly from DPD outputs of all questions in training data.
        forms = [x.strip() for x in open("tests/fixtures/data/wikitables/logical_forms_large_sample.txt")]
        expressions = self.world.process_sempre_forms(forms)
        for form, expression in zip(forms, expressions):
            action_sequence = self.world.get_action_sequence(expression)
            for action in action_sequence:
                assert "?" not in action, ("Found an unresolved type for form: %s\n"
                                           "Expression: %s\n"
                                           "Action sequence: %s\n" % (form, expression, action_sequence))
