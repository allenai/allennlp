# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.semparse import ParsingError
from allennlp.data.semparse.knowledge_graphs.table_knowledge_graph import TableKnowledgeGraph
from allennlp.data.semparse.worlds import WikiTablesWorld
from allennlp.data.tokenizers import Token


class TestWikiTablesWorldRepresentation(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.table_kg = TableKnowledgeGraph.read_from_file("tests/fixtures/data/wikitables/sample_table.tsv")
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '?']]
        self.world = WikiTablesWorld(self.table_kg, question_tokens)

    def test_world_processes_sempre_forms_correctly(self):
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world.parse_logical_form(sempre_form)
        # We add columns to the name mapping in sorted order, so "league" and "year" end up as C2
        # and C6.
        assert str(expression) == "R(C6,C2(cell:usl_a_league))"

    def test_world_parses_logical_forms_with_dates(self):
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'year', '2002', '?']]
        world = WikiTablesWorld(self.table_kg, question_tokens)
        sempre_form = "((reverse fb:row.row.league) (fb:row.row.year (fb:cell.cell.date (date 2002 -1 -1))))"
        expression = world.parse_logical_form(sempre_form)
        assert str(expression) == "R(C2,C6(D1(D0(2002,~1,~1))))"

    def test_world_parses_logical_forms_with_decimals(self):
        question_tokens = [Token(x) for x in ['0.2']]
        world = WikiTablesWorld(self.table_kg, question_tokens)
        sempre_form = "(fb:cell.cell.number (number 0.200))"
        expression = world.parse_logical_form(sempre_form)
        assert str(expression) == "I1(I(0_200))"

    def test_parsing_logical_forms_fails_with_unmapped_names(self):
        with pytest.raises(ParsingError):
            _ = self.world.parse_logical_form("(number 20)")

    def test_world_has_only_basic_numbers(self):
        valid_actions = self.world.get_valid_actions()
        for i in range(10):
            assert f'e -> {i}' in valid_actions['e']
        assert 'e -> 17' not in valid_actions['e']
        assert 'e -> 231' not in valid_actions['e']
        assert 'e -> 2007' not in valid_actions['e']
        assert 'e -> 2107' not in valid_actions['e']
        assert 'e -> 1800' not in valid_actions['e']

    def test_world_adds_numbers_from_question(self):
        question_tokens = [Token(x) for x in ['what', '2007', '2,107', '0.2', '1800s', '1950s', '?']]
        world = WikiTablesWorld(self.table_kg, question_tokens)
        valid_actions = world.get_valid_actions()
        assert 'e -> 2007' in valid_actions['e']
        assert 'e -> 2107' in valid_actions['e']

        # It appears that sempre normalizes floating point numbers.
        assert 'e -> 0.200' in valid_actions['e']

        # We want to add the end-points to things like "1800s": 1800 and 1900.
        assert 'e -> 1800' in valid_actions['e']
        assert 'e -> 1900' in valid_actions['e']
        assert 'e -> 1950' in valid_actions['e']
        assert 'e -> 1960' in valid_actions['e']

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

    def test_world_returns_correct_actions_with_lambda_with_var(self):
        sempre_form = ("((reverse fb:cell.cell.date) ((reverse fb:row.row.year) (argmax (number 1) "
                       "(number 1) (fb:row.row.league fb:cell.usl_a_league) (reverse (lambda x "
                       "((reverse fb:row.row.index) (var x)))))))")
        expression = self.world.parse_logical_form(sempre_form, remove_var_function=False)
        actions_with_var = self.world.get_action_sequence(expression)
        assert '<#1,#1> -> var' in actions_with_var
        assert 'r -> x' in actions_with_var

    def test_world_returns_correct_actions_with_lambda_without_var(self):
        sempre_form = ("((reverse fb:cell.cell.date) ((reverse fb:row.row.year) (argmax (number 1) "
                       "(number 1) (fb:row.row.league fb:cell.usl_a_league) (reverse (lambda x "
                       "((reverse fb:row.row.index) (var x)))))))")
        expression = self.world.parse_logical_form(sempre_form)
        actions_with_var = self.world.get_action_sequence(expression)
        assert '<#1,#1> -> var' not in actions_with_var
        assert 'r -> x' in actions_with_var

    @pytest.mark.skip(reason="fibonacci recursion currently going on here")
    def test_with_deeply_nested_logical_form(self):
        table_kg = TableKnowledgeGraph.read_from_file("tests/fixtures/data/wikitables/tables/109.tsv")
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'district', '?']]
        world = WikiTablesWorld(table_kg, question_tokens)
        logical_form = ("(count ((reverse fb:cell.cell.number) (or (or (or (or (or (or (or (or "
                        "(or (or (or (or (or (or (or (or (or (or (or (or (or fb:cell.virginia_1 "
                        "fb:cell.virginia_10) fb:cell.virginia_11) fb:cell.virginia_12) "
                        "fb:cell.virginia_13) fb:cell.virginia_14) fb:cell.virginia_15) "
                        "fb:cell.virginia_16) fb:cell.virginia_17) fb:cell.virginia_18) "
                        "fb:cell.virginia_19) fb:cell.virginia_2) fb:cell.virginia_20) "
                        "fb:cell.virginia_21) fb:cell.virginia_22) fb:cell.virginia_3) "
                        "fb:cell.virginia_4) fb:cell.virginia_5) fb:cell.virginia_6) "
                        "fb:cell.virginia_7) fb:cell.virginia_8) fb:cell.virginia_9)))")
        print("Parsing...")
        world.parse_logical_form(logical_form)
