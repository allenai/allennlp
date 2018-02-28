# pylint: disable=no-self-use,invalid-name
from typing import List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.semparse import ParsingError
from allennlp.data.semparse.knowledge_graphs import TableKnowledgeGraph
from allennlp.data.semparse.worlds import WikiTablesWorld
from allennlp.data.tokenizers import Token


def check_productions_match(actual_rules: List[str], expected_right_sides: List[str]):
    actual_right_sides = [rule.split(' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)


class TestWikiTablesWorldRepresentation(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.table_kg = TableKnowledgeGraph.read_from_file("tests/fixtures/data/wikitables/sample_table.tsv")
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '2000', '?']]
        self.world = WikiTablesWorld(self.table_kg, question_tokens)

    def test_get_valid_actions_returns_correct_set(self):
        # This test is long, but worth it.  These are all of the valid actions in the grammar, and
        # we want to be sure they are what we expect.
        valid_actions = self.world.get_valid_actions()
        assert set(valid_actions.keys()) == {
                '<#1,#1>',
                '<#1,<#1,#1>>',
                '<#1,d>',
                '<<#1,#2>,<#2,#1>>',
                '<d,<d,<#1,<<d,#1>,#1>>>>',
                '<d,<d,d>>',
                '<d,d>',
                '<d,e>',
                '<d,p>',
                '<d,r>',
                '<e,<e,<e,d>>>',
                '<e,d>',
                '<e,e>',
                '<e,p>',
                '<e,r>',
                '<p,d>',
                '<p,e>',
                '<p,p>',
                '<p,r>',
                '<r,d>',
                '<r,e>',
                '<r,p>',
                '<r,r>',
                '@START@',
                'd',
                'e',
                'p',
                'r',
                }

        check_productions_match(valid_actions['<#1,#1>'],
                                ['!=', 'fb:type.object.type'])

        check_productions_match(valid_actions['<#1,<#1,#1>>'],
                                ['and', 'or'])

        check_productions_match(valid_actions['<#1,d>'],
                                ['count'])

        check_productions_match(valid_actions['<<#1,#2>,<#2,#1>>'],
                                ['reverse'])

        check_productions_match(valid_actions['<d,<d,<#1,<<d,#1>,#1>>>>'],
                                ['argmax', 'argmin'])

        check_productions_match(valid_actions['<d,<d,d>>'],
                                ['-'])

        check_productions_match(valid_actions['<d,d>'],
                                ['<', '<=', '>', '>=', 'avg', 'min', 'sum', 'max',
                                 "['lambda x', d]", '[<<#1,#2>,<#2,#1>>, <d,d>]'])

        # These might look backwards, but that's because SEMPRE chose to make them backwards.
        # fb:a.b is a function that takes b and returns a.  So fb:cell.cell.date takes cell.date
        # and returns cell and fb:row.row.index takes row.index and returns row.
        check_productions_match(valid_actions['<d,e>'],
                                ['fb:cell.cell.num2', 'fb:cell.cell.date', 'fb:cell.cell.number',
                                 "['lambda x', e]", '[<<#1,#2>,<#2,#1>>, <e,d>]'])

        check_productions_match(valid_actions['<d,p>'],
                                ["['lambda x', p]", '[<<#1,#2>,<#2,#1>>, <p,d>]'])

        check_productions_match(valid_actions['<d,r>'],
                                ['fb:row.row.index', "['lambda x', r]", '[<<#1,#2>,<#2,#1>>, <r,d>]'])

        # "date" is a function that takes three numbers: (date 2018 01 06).
        check_productions_match(valid_actions['<e,<e,<e,d>>>'],
                                ['date'])

        check_productions_match(valid_actions['<e,d>'],
                                ['number', "['lambda x', d]", '[<<#1,#2>,<#2,#1>>, <d,e>]'])

        check_productions_match(valid_actions['<e,e>'],
                                ["['lambda x', e]", '[<<#1,#2>,<#2,#1>>, <e,e>]'])

        check_productions_match(valid_actions['<e,p>'],
                                ["['lambda x', p]", '[<<#1,#2>,<#2,#1>>, <p,e>]'])

        # Most of these are instance-specific production rules.  These are the columns in the
        # table.  Remember that SEMPRE did things backwards: fb:row.row.division takes a cell ID
        # and returns the row that has that cell in its row.division column.  This is why we have
        # to reverse all of these functions to go from a row to the cell in a particular column.
        check_productions_match(valid_actions['<e,r>'],
                                ['fb:row.row.null',  # This one is global, representing an empty set.
                                 'fb:row.row.year',
                                 'fb:row.row.league',
                                 'fb:row.row.avg_attendance',
                                 'fb:row.row.division',
                                 'fb:row.row.regular_season',
                                 'fb:row.row.playoffs',
                                 'fb:row.row.open_cup',
                                 "['lambda x', r]",
                                 '[<<#1,#2>,<#2,#1>>, <r,e>]'])

        # PART_TYPE rules.  A cell part is for when a cell has text that can be split into multiple
        # parts.  We don't currently handle this, so we don't have any terminal productions here.
        # We actually skip all logical forms that have "fb:part" productions, and we'll never
        # actually push one of these non-terminals onto our stack.  But they're in the grammar, so
        # we they are in our list of valid actions.
        check_productions_match(valid_actions['<p,d>'],
                                ["['lambda x', d]", '[<<#1,#2>,<#2,#1>>, <d,p>]'])

        check_productions_match(valid_actions['<p,e>'],
                                ['fb:cell.cell.part', "['lambda x', e]", '[<<#1,#2>,<#2,#1>>, <e,p>]'])

        check_productions_match(valid_actions['<p,p>'],
                                ["['lambda x', p]", '[<<#1,#2>,<#2,#1>>, <p,p>]'])

        check_productions_match(valid_actions['<p,r>'],
                                ["['lambda x', r]", '[<<#1,#2>,<#2,#1>>, <r,p>]'])

        check_productions_match(valid_actions['<r,d>'],
                                ["['lambda x', d]", '[<<#1,#2>,<#2,#1>>, <d,r>]'])

        check_productions_match(valid_actions['<r,e>'],
                                ["['lambda x', e]", '[<<#1,#2>,<#2,#1>>, <e,r>]'])

        check_productions_match(valid_actions['<r,p>'],
                                ["['lambda x', p]", '[<<#1,#2>,<#2,#1>>, <p,r>]'])

        check_productions_match(valid_actions['<r,r>'],
                                ['fb:row.row.next', "['lambda x', r]", '[<<#1,#2>,<#2,#1>>, <r,r>]'])

        check_productions_match(valid_actions['@START@'],
                                ['d', 'e', 'p', 'r'])

        check_productions_match(valid_actions['d'],
                                ['[<d,<d,d>>, d, d]',
                                 '[<#1,#1>, d]',
                                 '[<#1,<#1,#1>>, d, d]',
                                 '[<d,<d,<#1,<<d,#1>,#1>>>>, d, d, d, <d,d>]',
                                 '[<#1,d>, d]',
                                 '[<#1,d>, e]',
                                 '[<#1,d>, p]',
                                 '[<#1,d>, r]',
                                 '[<d,d>, d]',
                                 '[<e,d>, e]',
                                 '[<e,<e,<e,d>>>, e, e, e]',
                                 '[<r,d>, r]'])

        check_productions_match(valid_actions['e'],
                                ['-1',
                                 '0',
                                 '1',
                                 '2',
                                 '3',
                                 '4',
                                 '5',
                                 '6',
                                 '7',
                                 '8',
                                 '9',
                                 '2000',
                                 '[<#1,#1>, e]',
                                 '[<#1,<#1,#1>>, e, e]',
                                 '[<d,<d,<#1,<<d,#1>,#1>>>>, d, d, e, <d,e>]',
                                 '[<d,e>, d]',
                                 '[<p,e>, p]',
                                 '[<r,e>, r]',
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

        check_productions_match(valid_actions['p'],
                                ['[<d,<d,<#1,<<d,#1>,#1>>>>, d, d, p, <d,p>]',
                                 '[<#1,#1>, p]',
                                 '[<#1,<#1,#1>>, p, p]'])

        check_productions_match(valid_actions['r'],
                                ['fb:type.row',
                                 '[<#1,#1>, r]',
                                 '[<#1,<#1,#1>>, r, r]',
                                 '[<d,<d,<#1,<<d,#1>,#1>>>>, d, d, r, <d,r>]',
                                 '[<d,r>, d]',
                                 '[<e,r>, e]',
                                 '[<r,r>, r]'])

    def test_world_processes_sempre_forms_correctly(self):
        sempre_form = "((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world.parse_logical_form(sempre_form)
        # We add columns to the name mapping in sorted order, so "league" and "year" end up as C2
        # and C6.
        assert str(expression) == "R(C6,C2(cell:usl_a_league))"

    def test_world_parses_logical_forms_with_dates(self):
        sempre_form = "((reverse fb:row.row.league) (fb:row.row.year (fb:cell.cell.date (date 2000 -1 -1))))"
        expression = self.world.parse_logical_form(sempre_form)
        assert str(expression) == "R(C2,C6(D1(D0(2000,~1,~1))))"

    def test_world_parses_logical_forms_with_decimals(self):
        question_tokens = [Token(x) for x in ['0.2']]
        world = WikiTablesWorld(self.table_kg, question_tokens)
        sempre_form = "(fb:cell.cell.number (number 0.200))"
        expression = world.parse_logical_form(sempre_form)
        assert str(expression) == "I1(I(0_200))"

    def test_get_action_sequence_removes_currying_for_all_wikitables_functions(self):
        # minus
        logical_form = "(- (number 3) (number 2))"
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert 'd -> [<d,<d,d>>, d, d]' in action_sequence

        # date
        logical_form = "(count (fb:cell.cell.date (date 2000 -1 -1)))"
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert 'd -> [<e,<e,<e,d>>>, e, e, e]' in action_sequence

        # argmax
        logical_form = ("(argmax (number 1) (number 1) (fb:row.row.division fb:cell.2) "
                        "(reverse (lambda x ((reverse fb:row.row.index) (var x))))")
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert 'r -> [<d,<d,<#1,<<d,#1>,#1>>>>, d, d, r, <d,r>]' in action_sequence

        # and
        logical_form = "(and (number 1) (number 1))"
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert 'd -> [<#1,<#1,#1>>, d, d]' in action_sequence

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
        actions_without_var = self.world.get_action_sequence(expression)
        assert '<#1,#1> -> var' not in actions_without_var
        assert 'r -> x' in actions_without_var

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
