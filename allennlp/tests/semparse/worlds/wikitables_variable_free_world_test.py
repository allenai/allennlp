# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from typing import List

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph
from allennlp.semparse.worlds import WikiTablesVariableFreeWorld


def check_productions_match(actual_rules: List[str], expected_right_sides: List[str]):
    actual_right_sides = [rule.split(' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)


class TestWikiTablesVariableFreeWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '2013', '?']]
        self.table_file = self.FIXTURES_ROOT / 'data' / 'wikitables' / 'sample_table.tsv'
        self.table_kg = TableQuestionKnowledgeGraph.read_from_file(self.table_file, question_tokens)
        self.world = WikiTablesVariableFreeWorld(self.table_kg)
        table_file_with_date = self.FIXTURES_ROOT / 'data' / 'wikitables' / 'sample_table_with_date.tsv'
        table_kg_with_date = TableQuestionKnowledgeGraph.read_from_file(table_file_with_date, question_tokens)
        self.world_with_date = WikiTablesVariableFreeWorld(table_kg_with_date)

    def test_get_valid_actions_returns_correct_set(self):
        # This test is long, but worth it.  These are all of the valid actions in the grammar, and
        # we want to be sure they are what we expect.

        valid_actions = self.world.get_valid_actions()
        assert set(valid_actions.keys()) == {
                "<r,<l,s>>",
                "<r,<l,<n,r>>>",
                "<r,<l,r>>",
                "<r,<r,<l,n>>>",
                "<r,<l,<s,r>>>",
                "<n,<n,<n,d>>>",
                "<r,<l,<d,r>>>",
                "<r,<l,n>>",
                "<r,r>",
                "<r,n>",
                "d",
                "n",
                "s",
                "l",
                "r",
                "@start@",
                }

        check_productions_match(valid_actions['<r,<l,s>>'],
                                ['mode', 'select'])

        check_productions_match(valid_actions['<r,<l,<n,r>>>'],
                                ['filter_number_equals', 'filter_number_greater',
                                 'filter_number_greater_equals', 'filter_number_lesser',
                                 'filter_number_lesser_equals', 'filter_number_not_equals'])

        check_productions_match(valid_actions['<r,<l,r>>'],
                                ['argmax', 'argmin', 'same_as'])

        check_productions_match(valid_actions['<r,<r,<l,n>>>'],
                                ['diff'])

        check_productions_match(valid_actions['<r,<l,<s,r>>>'],
                                ['filter_in', 'filter_not_in'])

        check_productions_match(valid_actions['<n,<n,<n,d>>>'],
                                ['date'])

        check_productions_match(valid_actions['<r,<l,<d,r>>>'],
                                ['filter_date_equals', 'filter_date_greater',
                                 'filter_date_greater_equals', 'filter_date_lesser',
                                 'filter_date_lesser_equals', 'filter_date_not_equals'])

        check_productions_match(valid_actions['<r,<l,n>>'],
                                ['average', 'max', 'min', 'sum'])

        check_productions_match(valid_actions['<r,r>'],
                                ['first', 'last', 'next', 'previous'])

        check_productions_match(valid_actions['<r,n>'],
                                ['count'])

        # These are the columns in table, and are instance specific.
        check_productions_match(valid_actions['l'],
                                ['fb:row.row.year',
                                 'fb:row.row.league',
                                 'fb:row.row.avg_attendance',
                                 'fb:row.row.division',
                                 'fb:row.row.regular_season',
                                 'fb:row.row.playoffs',
                                 'fb:row.row.open_cup'])

        check_productions_match(valid_actions['@start@'],
                                ['d', 'n', 's'])

        # We merged cells and parts in SEMPRE to strings in this grammar.
        check_productions_match(valid_actions['s'],
                                ['fb:cell.2',
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
                                 'fb:cell.usl_first_division',
                                 'fb:part.4th',
                                 'fb:part.western',
                                 'fb:part.5th',
                                 '[<r,<l,s>>, r, l]'])

        check_productions_match(valid_actions['d'],
                                ['[<n,<n,<n,d>>>, n, n, n]'])

        check_productions_match(valid_actions['n'],
                                ['-1',
                                 '0',
                                 '1',
                                 '2013',
                                 '[<r,<l,n>>, r, l]',
                                 '[<r,<r,<l,n>>>, r, r, l]',
                                 '[<r,n>, r]'])

        check_productions_match(valid_actions['r'],
                                ['all_rows',
                                 '[<r,<l,<d,r>>>, r, l, d]',
                                 '[<r,<l,r>>, r, l]',
                                 '[<r,<l,<n,r>>>, r, l, n]',
                                 '[<r,<l,<s,r>>>, r, l, s]',
                                 '[<r,r>, r]'])

    def test_world_processes_logical_forms_correctly(self):
        logical_form = "(select (filter_in all_rows fb:row.row.league fb:cell.usl_a_league) fb:row.row.year)"
        expression = self.world.parse_logical_form(logical_form)
        # Cells (and parts) get mapped to strings.
        assert str(expression) == "S0(F30(R,C2,string:usl_a_league),C6)"

    def test_world_gets_correct_actions(self):
        logical_form = "(select (filter_in all_rows fb:row.row.league fb:cell.usl_a_league) fb:row.row.year)"
        expression = self.world.parse_logical_form(logical_form)
        expected_sequence = ['@start@ -> s', 's -> [<r,<l,s>>, r, l]', '<r,<l,s>> -> select',
                             'r -> [<r,<l,<s,r>>>, r, l, s]', '<r,<l,<s,r>>> -> filter_in',
                             'r -> all_rows', 'l -> fb:row.row.league', 's -> fb:cell.usl_a_league',
                             'l -> fb:row.row.year']
        assert self.world.get_action_sequence(expression) == expected_sequence

    def test_world_gets_logical_form_from_actions(self):
        logical_form = "(select (filter_in all_rows fb:row.row.league fb:cell.usl_a_league) fb:row.row.year)"
        expression = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(expression)
        reconstructed_logical_form = self.world.get_logical_form(action_sequence)
        assert logical_form == reconstructed_logical_form

    def test_world_processes_logical_forms_with_number_correctly(self):
        logical_form = "(select (filter_number_greater all_rows fb:row.row.year 2013) fb:row.row.year)"
        expression = self.world.parse_logical_form(logical_form)
        # Cells (and parts) get mapped to strings.
        assert str(expression) == "S0(F10(R,C6,num:2013),C6)"

    def test_world_processes_logical_forms_with_date_correctly(self):
        logical_form = "(select (filter_date_greater all_rows fb:row.row.year (date 2013 -1 -1)) fb:row.row.year)"
        expression = self.world.parse_logical_form(logical_form)
        # Cells (and parts) get mapped to strings.
        assert str(expression) == "S0(F20(R,C6,T0(num:2013,num:~1,num:~1)),C6)"

    def _get_world_with_question_tokens(self, tokens: List[Token]) -> WikiTablesVariableFreeWorld:
        table_kg = TableQuestionKnowledgeGraph.read_from_file(self.table_file, tokens)
        world = WikiTablesVariableFreeWorld(table_kg)
        return world

    def test_get_agenda(self):
        tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '2000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'n -> 2000',
                                           'l -> fb:row.row.year',
                                           '<r,<l,r>> -> argmax'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'difference', 'in', 'attendance',
                                     'between', 'years', '2001', 'and', '2005', '?']]
        world = self._get_world_with_question_tokens(tokens)
        # The agenda contains strings here instead of numbers because 2001 and 2005 actually link to
        # entities in the table whereas 2000 (in the previous case) does not.
        assert set(world.get_agenda()) == {'s -> fb:cell.2001',
                                           's -> fb:cell.2005',
                                           'l -> fb:row.row.year',
                                           '<r,<r,<l,n>>> -> diff'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'total', 'avg.', 'attendance', 'in',
                                     'years', '2001', 'and', '2005', '?']]
        world = self._get_world_with_question_tokens(tokens)
        # The agenda contains cells here instead of numbers because 2001 and 2005 actually link to
        # entities in the table whereas 2000 (in the previous case) does not.
        assert set(world.get_agenda()) == {'s -> fb:cell.2001',
                                           's -> fb:cell.2005',
                                           'l -> fb:row.row.year',
                                           'l -> fb:row.row.avg_attendance',
                                           '<r,<l,n>> -> sum'}
        tokens = [Token(x) for x in ['when', 'was', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'l -> fb:row.row.avg_attendance',
                                           '<r,<l,r>> -> argmin'
                                          }
        tokens = [Token(x) for x in ['what', 'is', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'l -> fb:row.row.avg_attendance',
                                           '<r,<l,n>> -> min'
                                          }

    def test_execute_works_with_select(self):
        logical_form = "(select all_rows fb:row.row.league)"
        cell_list = self.world.execute(logical_form)
        assert set(cell_list) == {'fb:cell.usl_a_league', 'fb:cell.usl_first_division'}

    def test_execute_works_with_argmax(self):
        logical_form = "(select (argmax all_rows fb:row.row.avg_attendance) fb:row.row.league)"
        cell_list = self.world.execute(logical_form)
        assert cell_list == ['fb:cell.usl_a_league']

    def test_execute_works_with_argmax_on_dates(self):
        logical_form = "(select (argmax all_rows fb:row.row.year) fb:row.row.league)"
        cell_list = self.world.execute(logical_form)
        assert cell_list == ['fb:cell.usl_first_division']

    def test_execute_works_with_argmin(self):
        logical_form = "(select (argmin all_rows fb:row.row.avg_attendance) fb:row.row.year)"
        cell_list = self.world.execute(logical_form)
        assert cell_list == ['fb:cell.2005']

    def test_execute_works_with_filter_number_greater(self):
        # Selecting cell values from all rows that have attendance greater than the min value of
        # attendance.
        logical_form = """(select (filter_number_greater all_rows fb:row.row.avg_attendance
                                   (min all_rows fb:row.row.avg_attendance)) fb:row.row.league)"""
        cell_value_list = self.world.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_a_league']

    def test_execute_works_with_filter_date_greater(self):
        # Selecting cell values from all rows that have date greater than 2002.
        logical_form = """(select (filter_date_greater all_rows fb:row.row.date
                                   (date 2002 -1 -1)) fb:row.row.league)"""
        cell_value_list = self.world_with_date.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_first_division']

    def test_execute_works_with_filter_number_greater_equals(self):
        # Counting rows that have attendance greater than or equal to the min value of attendance.
        logical_form = """(count (filter_number_greater_equals all_rows fb:row.row.avg_attendance
                                  (min all_rows fb:row.row.avg_attendance)))"""
        count_result = self.world.execute(logical_form)
        assert count_result == 2

    def test_execute_works_with_filter_date_greater_equals(self):
        # Selecting cell values from all rows that have date greater than or equal to 2005 February.
        logical_form = """(select (filter_date_greater_equals all_rows fb:row.row.date
                                   (date 2005 2 -1)) fb:row.row.league)"""
        cell_value_list = self.world_with_date.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_first_division']

    def test_execute_works_with_filter_number_lesser(self):
        # Selecting cell values from all rows that have date lesser than 2005.
        logical_form = """(select (filter_number_lesser all_rows fb:row.row.year 2005)
                           fb:row.row.league)"""
        cell_value_list = self.world.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_a_league']

    def test_execute_works_with_filter_date_lesser(self):
        # Selecting cell values from all rows that have date less that 2000 December
        logical_form = """(select (filter_date_lesser all_rows fb:row.row.date
                                   (date 2000 12 -1)) fb:row.row.league)"""
        cell_value_list = self.world_with_date.execute(logical_form)
        assert cell_value_list == []

    def test_execute_works_with_filter_number_lesser_equals(self):
        # Counting rows that have year lesser than or equal to 2005.
        logical_form = """(count (filter_number_lesser_equals all_rows fb:row.row.year 2005))"""
        count_result = self.world.execute(logical_form)
        assert count_result == 2

    def test_execute_works_with_filter_date_lesser_equals(self):
        # Selecting cell values from all rows that have date less that or equal to 2001 February 23
        logical_form = """(select (filter_date_lesser_equals all_rows fb:row.row.date
                                   (date 2001 2 23)) fb:row.row.league)"""
        cell_value_list = self.world_with_date.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_a_league']

    def test_execute_works_with_filter_number_equals(self):
        # Counting rows that have year equal to 2010.
        logical_form = """(count (filter_number_equals all_rows fb:row.row.year 2010))"""
        count_result = self.world.execute(logical_form)
        assert count_result == 0

    def test_execute_works_with_filter_date_equals(self):
        # Selecting cell values from all rows that have date not equal to 2001
        logical_form = """(select (filter_date_equals all_rows fb:row.row.date
                                   (date 2001 -1 -1)) fb:row.row.league)"""
        cell_value_list = self.world_with_date.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_a_league']

    def test_execute_works_with_filter_number_not_equals(self):
        # Counting rows that have year not equal to 2010.
        logical_form = """(count (filter_number_not_equals all_rows fb:row.row.year 2010))"""
        count_result = self.world.execute(logical_form)
        assert count_result == 2

    def test_execute_works_with_filter_date_not_equals(self):
        # Selecting cell values from all rows that have date not equal to 2001
        logical_form = """(select (filter_date_not_equals all_rows fb:row.row.date
                                   (date 2001 -1 -1)) fb:row.row.league)"""
        cell_value_list = self.world_with_date.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_first_division']

    def test_execute_works_with_filter_in(self):
        # Selecting "regular season" from rows that have "did not qualify" in "open cup" column.
        logical_form = """(select (filter_in all_rows fb:row.row.open_cup did_not_qualify)
                                  fb:row.row.regular_season)"""
        cell_list = self.world.execute(logical_form)
        assert cell_list == ["fb:cell.4th_western"]

    def test_execute_works_with_filter_not_in(self):
        # Selecting "regular season" from rows that do not have "did not qualify" in "open cup" column.
        logical_form = """(select (filter_not_in all_rows fb:row.row.open_cup did_not_qualify)
                                   fb:row.row.regular_season)"""
        cell_list = self.world.execute(logical_form)
        assert cell_list == ["fb:cell.5th"]

    def test_execute_works_with_first(self):
        # Selecting "regular season" from the first row.
        logical_form = """(select (first all_rows) fb:row.row.regular_season)"""
        cell_list = self.world.execute(logical_form)
        assert cell_list == ["fb:cell.4th_western"]

    def test_execute_works_with_last(self):
        # Selecting "regular season" from the last row where year is not equal to 2010.
        logical_form = """(select (last (filter_number_not_equals all_rows fb:row.row.year 2010))
                                  fb:row.row.regular_season)"""
        cell_list = self.world.execute(logical_form)
        assert cell_list == ["fb:cell.5th"]

    def test_execute_works_with_previous(self):
        # Selecting "regular season" from the row before last where year is not equal to 2010.
        logical_form = """(select (previous (last (filter_number_not_equals
                                                    all_rows fb:row.row.year 2010)))
                                  fb:row.row.regular_season)"""
        cell_list = self.world.execute(logical_form)
        assert cell_list == ["fb:cell.4th_western"]

    def test_execute_works_with_next(self):
        # Selecting "regular season" from the row after first where year is not equal to 2010.
        logical_form = """(select (next (first (filter_number_not_equals
                                                    all_rows fb:row.row.year 2010)))
                                  fb:row.row.regular_season)"""
        cell_list = self.world.execute(logical_form)
        assert cell_list == ["fb:cell.5th"]

    def test_execute_works_with_mode(self):
        # Most frequent division value.
        logical_form = """(mode all_rows fb:row.row.division)"""
        cell_list = self.world.execute(logical_form)
        assert cell_list == ["fb:cell.2"]
        # If we used selec instead, we should get a list of two values.
        logical_form = """(select all_rows fb:row.row.division)"""
        cell_list = self.world.execute(logical_form)
        assert cell_list == ["fb:cell.2", "fb:cell.2"]

    def test_execute_works_with_same_as(self):
        # Select the "league" from all the rows that have the same value under "playoffs" as the
        # row that has the string "a league" under "league".
        logical_form = """(select (same_as (filter_in all_rows fb:row.row.league a_league)
                                   fb:row.row.playoffs)
                           fb:row.row.league)"""
        cell_list = self.world.execute(logical_form)
        assert cell_list == ["fb:cell.usl_a_league", "fb:cell.usl_first_division"]

    def test_execute_works_with_sum(self):
        # Get total "avg attendance".
        logical_form = """(sum all_rows fb:row.row.avg_attendance)"""
        sum_value = self.world.execute(logical_form)
        assert sum_value == 13197
        # Total "avg attendance" where "playoffs" has "quarterfinals"
        logical_form = """(sum (filter_in all_rows fb:row.row.playoffs quarterfinals)
                                fb:row.row.avg_attendance)"""
        sum_value = self.world.execute(logical_form)
        assert sum_value == 13197

    def test_execute_works_with_average(self):
        # Get average "avg attendance".
        logical_form = """(average all_rows fb:row.row.avg_attendance)"""
        avg_value = self.world.execute(logical_form)
        assert avg_value == 6598.5
        # Average "avg attendance" where "playoffs" has "quarterfinals"
        logical_form = """(average (filter_in all_rows fb:row.row.playoffs quarterfinals)
                                fb:row.row.avg_attendance)"""
        avg_value = self.world.execute(logical_form)
        assert avg_value == 6598.5

    def test_execute_works_with_diff(self):
        # Difference in "avg attendance" between rows with "usl_a_league" and "usl_first_division"
        # in "league" columns.
        logical_form = """(diff (filter_in all_rows fb:row.row.league usl_a_league)
                                (filter_in all_rows fb:row.row.league usl_first_division)
                                fb:row.row.avg_attendance)"""
        avg_value = self.world.execute(logical_form)
        assert avg_value == 1141
