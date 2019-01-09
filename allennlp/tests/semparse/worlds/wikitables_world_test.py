# pylint: disable=no-self-use,invalid-name
from typing import List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.semparse import ParsingError
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph
from allennlp.semparse.worlds import WikiTablesWorld
from allennlp.semparse.type_declarations import wikitables_lambda_dcs as types


def check_productions_match(actual_rules: List[str], expected_right_sides: List[str]):
    actual_right_sides = [rule.split(' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)


class TestWikiTablesWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '2000', '?']]
        self.table_file = self.FIXTURES_ROOT / 'data' / 'wikitables' / 'sample_table.tsv'
        self.table_kg = TableQuestionKnowledgeGraph.read_from_file(self.table_file, question_tokens)
        self.world = WikiTablesWorld(self.table_kg)

    def test_get_valid_actions_returns_correct_set(self):
        # This test is long, but worth it.  These are all of the valid actions in the grammar, and
        # we want to be sure they are what we expect.

        # This test checks that our valid actions for each type match  PNP's, except for the
        # terminal productions for type 'p'.
        valid_actions = self.world.get_valid_actions()
        assert set(valid_actions.keys()) == {
                '<#1,#1>',
                '<#1,<#1,#1>>',
                '<#1,n>',
                '<<#1,#2>,<#2,#1>>',
                '<c,d>',
                '<c,n>',
                '<c,p>',
                '<c,r>',
                '<d,c>',
                '<d,d>',
                '<d,n>',
                '<d,r>',
                '<n,<n,<#1,<<#2,#1>,#1>>>>',
                '<n,<n,<n,d>>>',
                '<n,<n,n>>',
                '<n,c>',
                '<n,d>',
                '<n,n>',
                '<n,p>',
                '<n,r>',
                '<nd,nd>',
                '<p,c>',
                '<p,n>',
                '<r,c>',
                '<r,d>',
                '<r,n>',
                '<r,p>',
                '<r,r>',
                '@start@',
                'c',
                'd',
                'n',
                'p',
                'r',
                }

        check_productions_match(valid_actions['<#1,#1>'],
                                ['!='])

        check_productions_match(valid_actions['<#1,<#1,#1>>'],
                                ['and', 'or'])

        check_productions_match(valid_actions['<#1,n>'],
                                ['count'])

        check_productions_match(valid_actions['<<#1,#2>,<#2,#1>>'],
                                ['reverse'])

        check_productions_match(valid_actions['<c,d>'],
                                ["['lambda x', d]", '[<<#1,#2>,<#2,#1>>, <d,c>]'])

        check_productions_match(valid_actions['<c,n>'],
                                ["['lambda x', n]", '[<<#1,#2>,<#2,#1>>, <n,c>]'])

        check_productions_match(valid_actions['<c,p>'],
                                ['[<<#1,#2>,<#2,#1>>, <p,c>]'])

        # Most of these are instance-specific production rules.  These are the columns in the
        # table.  Remember that SEMPRE did things backwards: fb:row.row.division takes a cell ID
        # and returns the row that has that cell in its row.division column.  This is why we have
        # to reverse all of these functions to go from a row to the cell in a particular column.
        check_productions_match(valid_actions['<c,r>'],
                                ['fb:row.row.null',  # This one is global, representing an empty set.
                                 'fb:row.row.year',
                                 'fb:row.row.league',
                                 'fb:row.row.avg_attendance',
                                 'fb:row.row.division',
                                 'fb:row.row.regular_season',
                                 'fb:row.row.playoffs',
                                 'fb:row.row.open_cup'])

        # These might look backwards, but that's because SEMPRE chose to make them backwards.
        # fb:a.b is a function that takes b and returns a.  So fb:cell.cell.date takes cell.date
        # and returns cell and fb:row.row.index takes row.index and returns row.
        check_productions_match(valid_actions['<d,c>'],
                                ['fb:cell.cell.date',
                                 '[<<#1,#2>,<#2,#1>>, <c,d>]'])

        check_productions_match(valid_actions['<d,d>'],
                                ["['lambda x', d]", '[<<#1,#2>,<#2,#1>>, <d,d>]'])

        check_productions_match(valid_actions['<d,n>'],
                                ["['lambda x', n]", '[<<#1,#2>,<#2,#1>>, <n,d>]'])

        check_productions_match(valid_actions['<d,r>'],
                                ['[<<#1,#2>,<#2,#1>>, <r,d>]'])

        check_productions_match(valid_actions['<n,<n,<#1,<<#2,#1>,#1>>>>'],
                                ['argmax', 'argmin'])

        # "date" is a function that takes three numbers: (date 2018 01 06).
        check_productions_match(valid_actions['<n,<n,<n,d>>>'],
                                ['date'])

        check_productions_match(valid_actions['<n,<n,n>>'],
                                ['-'])

        check_productions_match(valid_actions['<n,c>'],
                                ['fb:cell.cell.num2', 'fb:cell.cell.number',
                                 '[<<#1,#2>,<#2,#1>>, <c,n>]'])

        check_productions_match(valid_actions['<n,d>'],
                                ["['lambda x', d]", '[<<#1,#2>,<#2,#1>>, <d,n>]'])

        check_productions_match(valid_actions['<n,n>'],
                                ['avg', 'sum', 'number',
                                 "['lambda x', n]", '[<<#1,#2>,<#2,#1>>, <n,n>]'])

        check_productions_match(valid_actions['<n,p>'],
                                ['[<<#1,#2>,<#2,#1>>, <p,n>]'])

        check_productions_match(valid_actions['<n,r>'],
                                ['fb:row.row.index', '[<<#1,#2>,<#2,#1>>, <r,n>]'])

        check_productions_match(valid_actions['<nd,nd>'],
                                ['<', '<=', '>', '>=', 'min', 'max'])

        # PART_TYPE rules.  A cell part is for when a cell has text that can be split into multiple
        # parts.
        check_productions_match(valid_actions['<p,c>'],
                                ['fb:cell.cell.part'])

        check_productions_match(valid_actions['<p,n>'],
                                ["['lambda x', n]"])

        check_productions_match(valid_actions['<r,c>'],
                                ['[<<#1,#2>,<#2,#1>>, <c,r>]'])

        check_productions_match(valid_actions['<r,d>'],
                                ["['lambda x', d]"])

        check_productions_match(valid_actions['<r,n>'],
                                ["['lambda x', n]", '[<<#1,#2>,<#2,#1>>, <n,r>]'])

        check_productions_match(valid_actions['<r,p>'],
                                ["['lambda x', p]", '[<<#1,#2>,<#2,#1>>, <p,r>]'])

        check_productions_match(valid_actions['<r,r>'],
                                ['fb:row.row.next', 'fb:type.object.type', '[<<#1,#2>,<#2,#1>>, <r,r>]'])

        check_productions_match(valid_actions['@start@'],
                                ['d', 'c', 'p', 'r', 'n'])

        check_productions_match(valid_actions['c'],
                                ['[<#1,#1>, c]',
                                 '[<#1,<#1,#1>>, c, c]',
                                 '[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, c, <n,c>]',
                                 '[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, c, <d,c>]',
                                 '[<d,c>, d]',
                                 '[<n,c>, n]',
                                 '[<p,c>, p]',
                                 '[<r,c>, r]',
                                 'fb:cell.null',
                                 'fb:cell.2',
                                 'fb:cell.2001',
                                 'fb:cell.2005',
                                 'fb:cell.4th_round',
                                 'fb:cell.4th_western',
                                 'fb:cell.5th',
                                 'fb:cell.6_028',
                                 'fb:cell.7_169',
                                 'fb:cell.did_not_qualify',
                                 'fb:cell.quarterfinals',
                                 'fb:cell.usl_a_league',
                                 'fb:cell.usl_first_division'])

        check_productions_match(valid_actions['d'],
                                ['[<n,<n,<n,d>>>, n, n, n]',
                                 '[<#1,#1>, d]',
                                 '[<#1,<#1,#1>>, d, d]',
                                 '[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, d, <d,d>]',
                                 '[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, d, <n,d>]',
                                 '[<c,d>, c]',
                                 '[<nd,nd>, d]'])

        check_productions_match(valid_actions['n'],
                                ['-1',
                                 '0',
                                 '1',
                                 '2000',
                                 '[<#1,#1>, n]',
                                 '[<#1,<#1,#1>>, n, n]',
                                 '[<#1,n>, c]',
                                 '[<#1,n>, d]',
                                 '[<#1,n>, n]',
                                 '[<#1,n>, p]',
                                 '[<#1,n>, r]',
                                 '[<c,n>, c]',
                                 '[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, n, <d,n>]',
                                 '[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, n, <n,n>]',
                                 '[<n,<n,n>>, n, n]',
                                 '[<n,n>, n]',
                                 '[<nd,nd>, n]',
                                 '[<r,n>, r]'])

        check_productions_match(valid_actions['p'],
                                ['[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, p, <n,p>]',
                                 '[<#1,#1>, p]',
                                 '[<c,p>, c]',
                                 '[<#1,<#1,#1>>, p, p]',
                                 'fb:part.4th',
                                 'fb:part.5th',
                                 'fb:part.western'])

        check_productions_match(valid_actions['r'],
                                ['fb:type.row',
                                 '[<#1,#1>, r]',
                                 '[<#1,<#1,#1>>, r, r]',
                                 '[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, r, <d,r>]',
                                 '[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, r, <n,r>]',
                                 '[<n,r>, n]',
                                 '[<c,r>, c]',
                                 '[<r,r>, r]'])

    def test_world_processes_sempre_forms_correctly(self):
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world.parse_logical_form(sempre_form)
        # We add columns to the name mapping in sorted order, so "league" and "year" end up as C2
        # and C6.
        f = types.name_mapper.get_alias
        assert str(expression) == f"{f('reverse')}(C6,C2(cell:usl_a_league))"

    def test_world_parses_logical_forms_with_dates(self):
        sempre_form = "((reverse fb:row.row.league) (fb:row.row.year (fb:cell.cell.date (date 2000 -1 -1))))"
        expression = self.world.parse_logical_form(sempre_form)
        f = types.name_mapper.get_alias
        assert str(expression) == \
                f"{f('reverse')}(C2,C6({f('fb:cell.cell.date')}({f('date')}(num:2000,num:~1,num:~1))))"

    def test_world_parses_logical_forms_with_decimals(self):
        question_tokens = [Token(x) for x in ['0.2']]
        table_kg = TableQuestionKnowledgeGraph.read_from_file(
                self.FIXTURES_ROOT / "data" / "wikitables" / "sample_table.tsv", question_tokens)
        world = WikiTablesWorld(table_kg)
        sempre_form = "(fb:cell.cell.number (number 0.200))"
        expression = world.parse_logical_form(sempre_form)
        f = types.name_mapper.get_alias
        assert str(expression) == f"{f('fb:cell.cell.number')}({f('number')}(num:0_200))"

    def test_get_action_sequence_removes_currying_for_all_wikitables_functions(self):
        # minus
        logical_form = "(- (number 0) (number 1))"
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert 'n -> [<n,<n,n>>, n, n]' in action_sequence

        # date
        logical_form = "(count (fb:cell.cell.date (date 2000 -1 -1)))"
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert 'd -> [<n,<n,<n,d>>>, n, n, n]' in action_sequence

        # argmax
        logical_form = ("(argmax (number 1) (number 1) (fb:row.row.division fb:cell.2) "
                        "(reverse (lambda x ((reverse fb:row.row.index) (var x)))))")
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert 'r -> [<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, r, <n,r>]' in action_sequence

        # and
        logical_form = "(and (number 1) (number 1))"
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert 'n -> [<#1,<#1,#1>>, n, n]' in action_sequence

    def test_parsing_logical_forms_fails_with_unmapped_names(self):
        with pytest.raises(ParsingError):
            _ = self.world.parse_logical_form("(number 20)")

    def test_world_has_only_basic_numbers(self):
        valid_actions = self.world.get_valid_actions()
        assert 'n -> -1' in valid_actions['n']
        assert 'n -> 0' in valid_actions['n']
        assert 'n -> 1' in valid_actions['n']
        assert 'n -> 17' not in valid_actions['n']
        assert 'n -> 231' not in valid_actions['n']
        assert 'n -> 2007' not in valid_actions['n']
        assert 'n -> 2107' not in valid_actions['n']
        assert 'n -> 1800' not in valid_actions['n']

    def test_world_adds_numbers_from_question(self):
        question_tokens = [Token(x) for x in ['what', '2007', '2,107', '0.2', '1800s', '1950s', '?']]
        table_kg = TableQuestionKnowledgeGraph.read_from_file(
                self.FIXTURES_ROOT / "data" / "wikitables" / "sample_table.tsv", question_tokens)
        world = WikiTablesWorld(table_kg)
        valid_actions = world.get_valid_actions()
        assert 'n -> 2007' in valid_actions['n']
        assert 'n -> 2107' in valid_actions['n']

        # It appears that sempre normalizes floating point numbers.
        assert 'n -> 0.200' in valid_actions['n']

        # We want to add the end-points to things like "1800s": 1800 and 1900.
        assert 'n -> 1800' in valid_actions['n']
        assert 'n -> 1900' in valid_actions['n']
        assert 'n -> 1950' in valid_actions['n']
        assert 'n -> 1960' in valid_actions['n']

    def test_world_returns_correct_actions_with_reverse(self):
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world.parse_logical_form(sempre_form)
        actions = self.world.get_action_sequence(expression)
        target_action_sequence = ['@start@ -> c', 'c -> [<r,c>, r]', '<r,c> -> [<<#1,#2>,<#2,#1>>, <c,r>]',
                                  '<<#1,#2>,<#2,#1>> -> reverse', '<c,r> -> fb:row.row.year',
                                  'r -> [<c,r>, c]', '<c,r> -> fb:row.row.league', 'c -> fb:cell.usl_a_league']
        assert actions == target_action_sequence

    def test_world_returns_correct_actions_with_two_reverses(self):
        sempre_form = ("(max ((reverse fb:cell.cell.date) ((reverse fb:row.row.year) "
                       "(fb:row.row.league fb:cell.usl_a_league))))")
        expression = self.world.parse_logical_form(sempre_form)
        actions = self.world.get_action_sequence(expression)
        target_action_sequence = ['@start@ -> d', 'd -> [<nd,nd>, d]', '<nd,nd> -> max', 'd -> [<c,d>, c]',
                                  '<c,d> -> [<<#1,#2>,<#2,#1>>, <d,c>]', '<<#1,#2>,<#2,#1>> -> reverse',
                                  '<d,c> -> fb:cell.cell.date', 'c -> [<r,c>, r]',
                                  '<r,c> -> [<<#1,#2>,<#2,#1>>, <c,r>]', '<<#1,#2>,<#2,#1>> -> reverse',
                                  '<c,r> -> fb:row.row.year', 'r -> [<c,r>, c]',
                                  '<c,r> -> fb:row.row.league', 'c -> fb:cell.usl_a_league']
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
        actions_without_var = self.world.get_action_sequence(expression)
        assert '<#1,#1> -> var' not in actions_without_var
        assert 'r -> x' in actions_without_var

    @pytest.mark.skip(reason="fibonacci recursion currently going on here")
    def test_with_deeply_nested_logical_form(self):
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'district', '?']]
        table_filename = self.FIXTURES_ROOT / 'data' / 'wikitables' / 'table' / '109.tsv'
        table_kg = TableQuestionKnowledgeGraph.read_from_file(table_filename, question_tokens)
        world = WikiTablesWorld(table_kg)
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

    def _get_world_with_question_tokens(self, tokens: List[Token]) -> WikiTablesWorld:
        table_kg = TableQuestionKnowledgeGraph.read_from_file(self.table_file, tokens)
        world = WikiTablesWorld(table_kg)
        return world

    def test_get_agenda(self):
        tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '2000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'n -> 2000',
                                           '<c,r> -> fb:row.row.year',
                                           '<n,<n,<#1,<<#2,#1>,#1>>>> -> argmax'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'difference', 'in', 'attendance',
                                     'between', 'years', '2001', 'and', '2005', '?']]
        world = self._get_world_with_question_tokens(tokens)
        # The agenda contains cells here instead of numbers because 2001 and 2005 actually link to
        # entities in the table whereas 2000 (in the previous case) does not.
        assert set(world.get_agenda()) == {'c -> fb:cell.2001',
                                           'c -> fb:cell.2005',
                                           '<c,r> -> fb:row.row.year',
                                           '<n,<n,n>> -> -'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'total', 'avg.', 'attendance', 'in',
                                     'years', '2001', 'and', '2005', '?']]
        world = self._get_world_with_question_tokens(tokens)
        # The agenda contains cells here instead of numbers because 2001 and 2005 actually link to
        # entities in the table whereas 2000 (in the previous case) does not.
        assert set(world.get_agenda()) == {'c -> fb:cell.2001',
                                           'c -> fb:cell.2005',
                                           '<c,r> -> fb:row.row.year',
                                           '<c,r> -> fb:row.row.avg_attendance',
                                           '<n,n> -> sum'}
        tokens = [Token(x) for x in ['when', 'was', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<c,r> -> fb:row.row.avg_attendance',
                                           '<n,<n,<#1,<<#2,#1>,#1>>>> -> argmin'
                                          }
        tokens = [Token(x) for x in ['what', 'is', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<c,r> -> fb:row.row.avg_attendance',
                                           '<nd,nd> -> min'
                                          }
