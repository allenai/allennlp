# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.semparse.knowledge_graphs.table_knowledge_graph import TableKnowledgeGraph
from allennlp.data.semparse.worlds import WikiTablesWorld
from allennlp.data.tokenizers import Token


class TestWikiTablesWorldRepresentation(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        table_kg = TableKnowledgeGraph.read_from_file("tests/fixtures/data/wikitables/sample_table.tsv")
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '?']]
        self.world = WikiTablesWorld(table_kg, question_tokens)

    def test_world_processes_sempre_forms_correctly(self):
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world.parse_logical_form(sempre_form)
        # We add columns to the name mapping in sorted order, so "league" and "year" end up as C2
        # and C6.
        assert str(expression) == "R(C6,C2(cell:usl_a_league))"

    def test_world_parses_logical_forms_with_dates(self):
        sempre_form = "((reverse fb:row.row.league) (fb:row.row.year (fb:cell.cell.date (date 2002 -1 -1))))"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "R(C2,C6(D1(D0(2002,~1,~1))))"

    def test_world_returns_correct_actions_with_reverse(self):
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world.parse_logical_form(sempre_form)
        actions = self.world.get_action_sequence(expression)
        target_action_sequence = ['@START@ -> e', 'e -> [<r,e>, r]', '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                                  '<<#1,#2>,<#2,#1>> -> reverse', '<e,r> -> fb:row.row.year',
                                  'r -> [<e,r>, e]', '<e,r> -> fb:row.row.league', 'e -> fb:cell.usl_a_league']
        assert actions == target_action_sequence

    def test_world_returns_correct_actions_with_two_reverses(self):
        sempre_form = ("(max ((reverse fb:cell.cell.date) ((reverse fb:row.row.year) "
                       "(fb:row.row.league fb:cell.usl_a_league))))")
        expression = self.world.parse_logical_form(sempre_form)
        actions = self.world.get_action_sequence(expression)
        target_action_sequence = ['@START@ -> d', 'd -> [<d,d>, d]', '<d,d> -> max', 'd -> [<e,d>, e]',
                                  '<e,d> -> [<<#1,#2>,<#2,#1>>, <d,e>]', '<<#1,#2>,<#2,#1>> -> reverse',
                                  '<d,e> -> fb:cell.cell.date', 'e -> [<r,e>, r]',
                                  '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]', '<<#1,#2>,<#2,#1>> -> reverse',
                                  '<e,r> -> fb:row.row.year', 'r -> [<e,r>, e]',
                                  '<e,r> -> fb:row.row.league', 'e -> fb:cell.usl_a_league']
        assert actions == target_action_sequence

    def test_large_scale_processing(self):
        # A sample of 500 logical forms taken randomly from DPD outputs of all questions in training data.
        forms = [x.strip() for x in open("tests/fixtures/data/wikitables/logical_forms_large_sample.txt")]
        expressions = [self.world.parse_logical_form(form) for form in forms]
        for form, expression in zip(forms, expressions):
            action_sequence = self.world.get_action_sequence(expression)
            for action in action_sequence:
                assert "?" not in action, ("Found an unresolved type for form: %s\n"
                                           "Expression: %s\n"
                                           "Action sequence: %s\n" % (form, expression, action_sequence))
