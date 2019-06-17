# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from typing import List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import WordTokenizer
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse import ExecutionError
from allennlp.semparse.common import Date
from allennlp.semparse.domain_languages.wikitables_language import WikiTablesLanguage
from allennlp.tests.semparse.domain_languages.domain_language_test import check_productions_match


class TestWikiTablesLanguage(AllenNlpTestCase):
    # TODO(mattg, pradeep): Add tests for the ActionSpaceWalker as well.
    def setUp(self):
        super().setUp()
        # Adding a bunch of random tokens in here so we get them as constants in the language.
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '2013', '?',
                                              'quarterfinals', 'a_league', '2010', '8000',
                                              'did_not_qualify', '2001', '2', '23', '2005', '1',
                                              '2002', 'usl_a_league', 'usl_first_division']]
        self.table_file = self.FIXTURES_ROOT / 'data' / 'wikitables' / 'sample_table.tagged'
        self.table_context = TableQuestionContext.read_from_file(self.table_file, question_tokens)
        self.language = WikiTablesLanguage(self.table_context)

    def _get_world_with_question_tokens(self, tokens: List[Token]) -> WikiTablesLanguage:
        table_context = TableQuestionContext.read_from_file(self.table_file, tokens)
        world = WikiTablesLanguage(table_context)
        return world

    def _get_world_with_question_tokens_and_table_file(self,
                                                       tokens: List[Token],
                                                       table_file: str) -> WikiTablesLanguage:
        table_context = TableQuestionContext.read_from_file(table_file, tokens)
        world = WikiTablesLanguage(table_context)
        return world

    def test_execute_fails_with_unknown_function(self):
        logical_form = "(unknown_function all_rows string_column:league)"
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_select(self):
        logical_form = "(select_string all_rows string_column:league)"
        cell_list = self.language.execute(logical_form)
        assert set(cell_list) == {'usl_a_league', 'usl_first_division'}

    def test_execute_works_with_select_number(self):
        logical_form = "(select_number all_rows number_column:division)"
        selected_number = self.language.execute(logical_form)
        assert selected_number == 2.0

    def test_execute_works_with_argmax(self):
        logical_form = "(select_string (argmax all_rows number_column:avg_attendance) string_column:league)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == ['usl_a_league']

    def test_execute_works_with_argmax_on_dates(self):
        logical_form = "(select_string (argmax all_rows date_column:year) string_column:league)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == ['usl_first_division']

    def test_execute_works_with_argmin(self):
        logical_form = "(select_date (argmin all_rows number_column:avg_attendance) date_column:year)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == Date(2005, 3, -1)

    def test_execute_works_with_argmin_on_dates(self):
        logical_form = "(select_string (argmin all_rows date_column:year) string_column:league)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == ['usl_a_league']

    def test_execute_works_with_filter_number_greater(self):
        # Selecting cell values from all rows that have attendance greater than the min value of
        # attendance.
        logical_form = """(select_string (filter_number_greater all_rows number_column:avg_attendance
                                   (min_number all_rows number_column:avg_attendance)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_number_greater all_rows number_column:avg_attendance
                                   all_rows) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_number_greater all_rows number_column:avg_attendance
                            string:usl_first_division) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_date_greater(self):
        # Selecting cell values from all rows that have date greater than 2002.
        logical_form = """(select_string (filter_date_greater all_rows date_column:year
                                   (date 2002 -1 -1)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_greater all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_number_greater_equals(self):
        # Counting rows that have attendance greater than or equal to the min value of attendance.
        logical_form = """(count (filter_number_greater_equals all_rows number_column:avg_attendance
                                  (min_number all_rows number_column:avg_attendance)))"""
        count_result = self.language.execute(logical_form)
        assert count_result == 2
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_greater all_rows number_column:avg_attendance all_rows))"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_greater all_rows number_column:avg_attendance
                                  string:usl_a_league))"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_date_greater_equals(self):
        # Selecting cell values from all rows that have date greater than or equal to 2005 February
        # 1st.
        logical_form = """(select_string (filter_date_greater_equals all_rows date_column:year
                                   (date 2005 2 1)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_greater_equals all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_number_lesser(self):
        # Selecting cell values from all rows that have date lesser than 2005.
        logical_form = """(select_string (filter_number_lesser all_rows number_column:avg_attendance
                                    (max_number all_rows number_column:avg_attendance)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_first_division']

    def test_execute_works_with_filter_date_lesser(self):
        # Selecting cell values from all rows that have date less that 2005 January
        logical_form = """(select_string (filter_date_lesser all_rows date_column:year
                                   (date 2005 1 -1)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ["usl_a_league"]
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_lesser all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_number_lesser_equals(self):
        # Counting rows that have year lesser than or equal to 2005.
        logical_form = """(count (filter_number_lesser_equals all_rows number_column:avg_attendance 8000))"""
        count_result = self.language.execute(logical_form)
        assert count_result == 2

    def test_execute_works_with_filter_date_lesser_equals(self):
        # Selecting cell values from all rows that have date less that or equal to 2001 February 23
        logical_form = """(select_string (filter_date_lesser_equals all_rows date_column:year
                                   (date 2001 2 23)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_lesser_equals all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_number_equals(self):
        # Counting rows that have year equal to 2010.
        logical_form = """(count (filter_number_equals all_rows number_column:avg_attendance 8000))"""
        count_result = self.language.execute(logical_form)
        assert count_result == 0

    def test_execute_works_with_filter_date_equals(self):
        # Selecting cell values from all rows that have date not equal to 2001
        logical_form = """(select_string (filter_date_equals all_rows date_column:year
                                   (date 2001 -1 -1)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_equals all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_number_not_equals(self):
        # Counting rows that have year not equal to 2010.
        logical_form = """(count (filter_number_not_equals all_rows number_column:avg_attendance 8000))"""
        count_result = self.language.execute(logical_form)
        assert count_result == 2

    def test_execute_works_with_filter_date_not_equals(self):
        # Selecting cell values from all rows that have date not equal to 2001
        logical_form = """(select_string (filter_date_not_equals all_rows date_column:year
                                   (date 2001 -1 -1)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_not_equals all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_in(self):
        # Selecting "regular season" from rows that have "did not qualify" in "open cup" column.
        logical_form = """(select_string (filter_in all_rows string_column:open_cup string:did_not_qualify)
                                  string_column:regular_season)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["4th_western"]

    def test_execute_works_with_select_nested_in_filter_in(self):
        logical_form = """(filter_in all_rows string_column:regular_season (select_string (first all_rows)
                           string_column:regular_season))"""
        row_list = self.language.execute(logical_form)
        assert row_list == self.language.execute("(first all_rows)")

    def test_execute_works_with_filter_not_in(self):
        # Selecting "regular season" from rows that do not have "did not qualify" in "open cup" column.
        logical_form = """(select_string (filter_not_in all_rows string_column:open_cup string:did_not_qualify)
                                   string_column:regular_season)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["5th"]
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_not_in all_rows string_column:open_cup 2000)
                                   string_column:regular_season)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_first(self):
        # Selecting "regular season" from the first row.
        logical_form = """(select_string (first all_rows) string_column:regular_season)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["4th_western"]

    def test_execute_logs_warning_with_first_on_empty_list(self):
        # Selecting "regular season" from the first row where year is greater than 2010.
        with self.assertLogs("allennlp.semparse.domain_languages.wikitables_language") as log:
            logical_form = """(select_string (first (filter_date_greater all_rows date_column:year
                                                (date 2010 -1 -1)))
                                      string_column:regular_season)"""
            self.language.execute(logical_form)
        self.assertEqual(log.output,
                         ["WARNING:allennlp.semparse.domain_languages.wikitables_language:"
                          "Trying to get first row from an empty list"])

    def test_execute_works_with_last(self):
        # Selecting "regular season" from the last row where year is not equal to 2010.
        logical_form = """(select_string (last (filter_date_not_equals all_rows date_column:year
                                         (date 2010 -1 -1)))
                                  string_column:regular_season)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["5th"]

    def test_execute_logs_warning_with_last_on_empty_list(self):
        # Selecting "regular season" from the last row where year is greater than 2010.
        with self.assertLogs("allennlp.semparse.domain_languages.wikitables_language") as log:
            logical_form = """(select_string (last (filter_date_greater all_rows date_column:year
                                                (date 2010 -1 -1)))
                                      string_column:regular_season)"""
            self.language.execute(logical_form)
        self.assertEqual(log.output,
                         ["WARNING:allennlp.semparse.domain_languages.wikitables_language:"
                          "Trying to get last row from an empty list"])

    def test_execute_works_with_previous(self):
        # Selecting "regular season" from the row before last where year is not equal to 2010.
        logical_form = """(select_string (previous (last (filter_date_not_equals
                                                    all_rows date_column:year (date 2010 -1 -1))))
                                  string_column:regular_season)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["4th_western"]

    def test_execute_works_with_next(self):
        # Selecting "regular season" from the row after first where year is not equal to 2010.
        logical_form = """(select_string (next (first (filter_date_not_equals
                                                all_rows date_column:year (date 2010 -1 -1))))
                                  string_column:regular_season)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["5th"]

    def test_execute_works_with_max_date(self):
        logical_form = """(max_date all_rows date_column:year)"""
        cell_list = self.language.execute(logical_form)
        assert str(cell_list) == "2005"

    def test_execute_works_with_min_date(self):
        logical_form = """(min_date all_rows date_column:year)"""
        cell_list = self.language.execute(logical_form)
        assert str(cell_list) == "2001"

    def test_execute_works_with_mode_number(self):
        # Most frequent division value.
        logical_form = """(mode_number all_rows number_column:division)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == 2.0
        logical_form = """(mode_number
                            (filter_in all_rows string_column:league string:a_league)
                           number_column:division)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == 2.0

    def test_execute_works_with_mode_string(self):
        logical_form = """(mode_string all_rows string_column:league)"""
        cell_list = self.language.execute(logical_form)
        # Returns the string values with frequency 1 (which is the max frequency)
        assert cell_list == ["usl_a_league", "usl_first_division"]

    def test_execute_works_with_mode_date(self):
        logical_form = """(mode_date all_rows date_column:year)"""
        cell_list = self.language.execute(logical_form)
        assert str(cell_list) == "2001"

    def test_execute_works_with_same_as(self):
        # Select the "league" from all the rows that have the same value under "playoffs" as the
        # row that has the string "a league" under "league".
        logical_form = """(select_string (same_as (filter_in all_rows string_column:league string:a_league)
                                   string_column:playoffs)
                           string_column:league)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["usl_a_league", "usl_first_division"]

    def test_execute_works_with_sum(self):
        # Get total "avg attendance".
        logical_form = """(sum all_rows number_column:avg_attendance)"""
        sum_value = self.language.execute(logical_form)
        assert sum_value == 13197
        # Total "avg attendance" where "playoffs" has "quarterfinals"
        logical_form = """(sum (filter_in all_rows string_column:playoffs string:quarterfinals)
                                number_column:avg_attendance)"""
        sum_value = self.language.execute(logical_form)
        assert sum_value == 13197

    def test_execute_works_with_average(self):
        # Get average "avg attendance".
        logical_form = """(average all_rows number_column:avg_attendance)"""
        avg_value = self.language.execute(logical_form)
        assert avg_value == 6598.5
        # Average "avg attendance" where "playoffs" has "quarterfinals"
        logical_form = """(average (filter_in all_rows string_column:playoffs string:quarterfinals)
                                number_column:avg_attendance)"""
        avg_value = self.language.execute(logical_form)
        assert avg_value == 6598.5

    def test_execute_works_with_diff(self):
        # Difference in "avg attendance" between rows with "usl_a_league" and "usl_first_division"
        # in "league" columns.
        logical_form = """(diff (filter_in all_rows string_column:league string:usl_a_league)
                                (filter_in all_rows string_column:league string:usl_first_division)
                                number_column:avg_attendance)"""
        avg_value = self.language.execute(logical_form)
        assert avg_value == 1141

    def test_execute_fails_with_diff_on_non_numerical_columns(self):
        logical_form = """(diff (filter_in all_rows string_column:league string:usl_a_league)
                                (filter_in all_rows string_column:league string:usl_first_division)
                                string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_number_comparison_works(self):
        # TableQuestionContext normlaizes all strings according to some rules. We want to ensure
        # that the original numerical values of number cells is being correctly processed here.
        tokens = WordTokenizer().tokenize("when was the attendance the highest?")
        tagged_file = self.FIXTURES_ROOT / "data" / "corenlp_processed_tables" / "TEST-2.table"
        language = self._get_world_with_question_tokens_and_table_file(tokens, tagged_file)
        result = language.execute("(select_date (argmax all_rows number_column:attendance) date_column:date)")
        assert result == Date(-1, 11, 10)

    def test_evaluate_logical_form(self):
        logical_form = """(select_string (same_as (filter_in all_rows string_column:league string:a_league)
                                   string_column:playoffs)
                           string_column:league)"""
        assert self.language.evaluate_logical_form(logical_form, ["USL A-League",
                                                                  "USL First Division"])

    def test_evaluate_logical_form_with_invalid_logical_form(self):
        logical_form = """(select_string (same_as (filter_in all_rows string_column:league INVALID_CONSTANT)
                                   string_column:playoffs)
                           string_column:league)"""
        assert not self.language.evaluate_logical_form(logical_form, ["USL A-League",
                                                                      "USL First Division"])

    def test_get_nonterminal_productions_all_column_types(self):
        # This test is long, but worth it.  These are all of the valid actions in the grammar, and
        # we want to be sure they are what we expect.
        productions = self.language.get_nonterminal_productions()
        assert set(productions.keys()) == {
                "@start@",
                "<List[Row],StringColumn:List[str]>",
                "<List[Row],DateColumn:Date>",
                "<List[Row],NumberColumn,Number:List[Row]>",
                "<List[Row],ComparableColumn:List[Row]>",
                "<List[Row],Column:List[Row]>",
                "<List[Row],List[Row],NumberColumn:Number>",
                "<List[Row],StringColumn,List[str]:List[Row]>",
                "<Number,Number,Number:Date>",
                "<List[Row],DateColumn,Date:List[Row]>",
                "<List[Row],NumberColumn:Number>",
                "<List[Row]:List[Row]>",
                '<List[Row],StringColumn:List[str]>',
                "<List[Row]:Number>",
                "List[str]",
                "List[Row]",
                "Date",
                "Number",
                "StringColumn",
                "NumberColumn",
                "ComparableColumn",
                "Column",
                "DateColumn",
                "List[str]",
                }

        check_productions_match(productions['@start@'],
                                ['Date', 'Number', 'List[str]'])

        check_productions_match(productions['<List[Row],StringColumn:List[str]>'],
                                ['select_string', 'mode_string'])

        check_productions_match(productions['<List[Row],DateColumn:Date>'],
                                ['select_date', 'max_date', 'min_date', 'mode_date'])

        check_productions_match(productions['<List[Row],NumberColumn,Number:List[Row]>'],
                                ['filter_number_equals', 'filter_number_greater',
                                 'filter_number_greater_equals', 'filter_number_lesser',
                                 'filter_number_lesser_equals', 'filter_number_not_equals'])

        check_productions_match(productions['<List[Row],ComparableColumn:List[Row]>'],
                                ['argmax', 'argmin'])

        check_productions_match(productions['<List[Row],Column:List[Row]>'],
                                ['same_as'])

        check_productions_match(productions['<List[Row],List[Row],NumberColumn:Number>'],
                                ['diff'])

        check_productions_match(productions['<List[Row],StringColumn,List[str]:List[Row]>'],
                                ['filter_in', 'filter_not_in'])

        check_productions_match(productions['<Number,Number,Number:Date>'],
                                ['date'])

        check_productions_match(productions['<List[Row],DateColumn,Date:List[Row]>'],
                                ['filter_date_equals', 'filter_date_greater',
                                 'filter_date_greater_equals', 'filter_date_lesser',
                                 'filter_date_lesser_equals', 'filter_date_not_equals'])

        check_productions_match(productions['<List[Row],NumberColumn:Number>'],
                                ['average', 'max_number', 'min_number', 'sum',
                                 'select_number', 'mode_number'])

        check_productions_match(productions['<List[Row]:List[Row]>'],
                                ['first', 'last', 'next', 'previous'])

        check_productions_match(productions['<List[Row]:Number>'],
                                ['count'])

        check_productions_match(productions['List[Row]'],
                                ['all_rows',
                                 '[<List[Row],Column:List[Row]>, List[Row], Column]',
                                 '[<List[Row],DateColumn,Date:List[Row]>, List[Row], DateColumn, Date]',
                                 '[<List[Row],ComparableColumn:List[Row]>, List[Row], ComparableColumn]',
                                 '[<List[Row],NumberColumn,Number:List[Row]>, List[Row], NumberColumn, Number]',
                                 '[<List[Row],StringColumn,List[str]:List[Row]>, List[Row], StringColumn, List[str]]',  # pylint: disable=line-too-long
                                 '[<List[Row]:List[Row]>, List[Row]]'])

        check_productions_match(productions['Date'],
                                ['[<Number,Number,Number:Date>, Number, Number, Number]',
                                 '[<List[Row],DateColumn:Date>, List[Row], DateColumn]'])

        # Some of the number productions are instance-specific, and some of them are from the
        # grammar.
        check_productions_match(productions['Number'],
                                ['2001',
                                 '2002',
                                 '2005',
                                 '2010',
                                 '2013',
                                 '-1',
                                 '1',
                                 '2',
                                 '23',
                                 '8000',
                                 '[<List[Row],NumberColumn:Number>, List[Row], NumberColumn]',
                                 '[<List[Row],List[Row],NumberColumn:Number>, List[Row], List[Row], NumberColumn]',
                                 '[<List[Row]:Number>, List[Row]]'])

        # These are the columns in table, and are instance specific.
        check_productions_match(productions['StringColumn'],
                                ['string_column:league',
                                 'string_column:playoffs',
                                 'string_column:open_cup',
                                 'string_column:year',
                                 'string_column:division',
                                 'string_column:avg_attendance',
                                 'string_column:regular_season'])

        check_productions_match(productions['DateColumn'],
                                ['date_column:year'])

        check_productions_match(productions['NumberColumn'],
                                ['number_column:avg_attendance',
                                 'number_column:open_cup',
                                 'number_column:regular_season',
                                 'number_column:division',
                                 'number_column:year'])

        check_productions_match(productions['ComparableColumn'],
                                ['date_column:year',
                                 'number_column:avg_attendance',
                                 'number_column:open_cup',
                                 'number_column:regular_season',
                                 'number_column:division',
                                 'number_column:year'])

        check_productions_match(productions['Column'],
                                ['string_column:league',
                                 'string_column:playoffs',
                                 'string_column:open_cup',
                                 'string_column:year',
                                 'string_column:division',
                                 'string_column:avg_attendance',
                                 'string_column:regular_season',
                                 'date_column:year',
                                 'number_column:avg_attendance',
                                 'number_column:open_cup',
                                 'number_column:regular_season',
                                 'number_column:division',
                                 'number_column:year'])

        # Strings come from the question - any span in the question that shows up as a cell in the
        # table is a valid string production.
        check_productions_match(productions['List[str]'],
                                ['string:quarterfinals',
                                 'string:did_not_qualify',
                                 'string:a_league',
                                 'string:usl_first_division',
                                 'string:usl_a_league',
                                 'string:1',
                                 'string:2',
                                 'string:2005',
                                 'string:2001',
                                 '[<List[Row],StringColumn:List[str]>, List[Row], StringColumn]'])

    def test_world_processes_logical_forms_correctly(self):
        logical_form = ("(select_date (filter_in all_rows string_column:league string:usl_a_league)"
                        " date_column:year)")
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.action_sequence_to_logical_form(action_sequence) == logical_form

    def test_world_gets_correct_actions(self):
        logical_form = """(select_date (filter_in all_rows string_column:league string:usl_a_league)
                           date_column:year)"""
        expected_sequence = ['@start@ -> Date',
                             'Date -> [<List[Row],DateColumn:Date>, List[Row], DateColumn]',
                             '<List[Row],DateColumn:Date> -> select_date',
                             'List[Row] -> [<List[Row],StringColumn,List[str]:List[Row]>, '
                                     'List[Row], StringColumn, List[str]]',  # pylint: disable=bad-continuation
                             '<List[Row],StringColumn,List[str]:List[Row]> -> filter_in',
                             'List[Row] -> all_rows',
                             'StringColumn -> string_column:league',
                             'List[str] -> string:usl_a_league',
                             'DateColumn -> date_column:year']
        assert self.language.logical_form_to_action_sequence(logical_form) == expected_sequence

    def test_world_processes_logical_forms_with_number_correctly(self):
        logical_form = ("(select_date (filter_number_greater all_rows number_column:avg_attendance 8000) "
                        "date_column:year)")
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.action_sequence_to_logical_form(action_sequence) == logical_form

    def test_world_processes_logical_forms_with_date_correctly(self):
        logical_form = ("(select_date (filter_date_greater all_rows date_column:year (date 2013 -1 -1)) "
                        "date_column:year)")
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.action_sequence_to_logical_form(action_sequence) == logical_form

    def test_world_processes_logical_forms_with_generic_function_correctly(self):
        logical_form = ("(select_string (argmax all_rows date_column:year) string_column:league)")
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.action_sequence_to_logical_form(action_sequence) == logical_form

    def test_get_agenda(self):
        tokens = [Token(x) for x in ['what', 'was', 'the', 'difference', 'in', 'attendance',
                                     'between', 'years', '2001', 'and', '2005', '?']]
        world = self._get_world_with_question_tokens(tokens)
        # "year" column does not match because "years" occurs in the question.
        assert set(world.get_agenda()) == {'Number -> 2001',
                                           'Number -> 2005',
                                           'List[str] -> string:2005',
                                           'List[str] -> string:2001',
                                           '<List[Row],DateColumn,Date:List[Row]> -> filter_date_equals',
                                           '<List[Row],List[Row],NumberColumn:Number> -> diff'}
        # Conservative agenda does not have strings and numbers because they have multiple types.
        assert set(world.get_agenda(conservative=True)) == {
                '<List[Row],List[Row],NumberColumn:Number> -> diff',
                '<List[Row],DateColumn,Date:List[Row]> -> filter_date_equals'}

        tokens = [Token(x) for x in ['what', 'was', 'the', 'total', 'avg.', 'attendance', 'in',
                                     'years', '2001', 'and', '2005', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'Number -> 2001',
                                           'Number -> 2005',
                                           'List[str] -> string:2005',
                                           'List[str] -> string:2001',
                                           '<List[Row],NumberColumn:Number> -> sum',
                                           '<List[Row],DateColumn,Date:List[Row]> -> filter_date_equals',
                                           'StringColumn -> string_column:avg_attendance',
                                           'NumberColumn -> number_column:avg_attendance'}
        # Conservative disallows "sum" for the question word "total" too.
        assert set(world.get_agenda(conservative=True)) == {
                '<List[Row],DateColumn,Date:List[Row]> -> filter_date_equals'}

        tokens = [Token(x) for x in ['what', 'was', 'the', 'average', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],NumberColumn:Number> -> average',
                                           'StringColumn -> string_column:avg_attendance',
                                           'NumberColumn -> number_column:avg_attendance'}
        assert set(world.get_agenda(conservative=True)) == {'<List[Row],NumberColumn:Number> -> average'}

        tokens = [Token(x) for x in ['what', 'was', 'the', 'largest', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],ComparableColumn:List[Row]> -> argmax',
                                           'StringColumn -> string_column:avg_attendance',
                                           'NumberColumn -> number_column:avg_attendance'}
        assert set(world.get_agenda(conservative=True)) == {'<List[Row],ComparableColumn:List[Row]> -> argmax'}

        tokens = [Token(x) for x in ['when', 'was', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],ComparableColumn:List[Row]> -> argmin',
                                           'StringColumn -> string_column:avg_attendance',
                                           '<List[Row],DateColumn:Date> -> select_date',
                                           'NumberColumn -> number_column:avg_attendance'}
        assert set(world.get_agenda(conservative=True)) == {'<List[Row],ComparableColumn:List[Row]> -> argmin',
                                                            '<List[Row],DateColumn:Date> -> select_date'}

        tokens = [Token(x) for x in ['what', 'was', 'the', 'attendance', 'after', 'the',
                                     'time', 'with', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],ComparableColumn:List[Row]> -> argmin',
                                           'StringColumn -> string_column:avg_attendance',
                                           '<List[Row]:List[Row]> -> next',
                                           'NumberColumn -> number_column:avg_attendance'}
        # conservative disallows "after" mapping to "next"
        assert set(world.get_agenda(conservative=True)) == {'<List[Row],ComparableColumn:List[Row]> -> argmin'}

        tokens = [Token(x) for x in ['what', 'was', 'the', 'attendance', 'below', 'the',
                                     'row', 'with', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],ComparableColumn:List[Row]> -> argmin',
                                           'StringColumn -> string_column:avg_attendance',
                                           '<List[Row]:List[Row]> -> next',
                                           'NumberColumn -> number_column:avg_attendance'}
        assert set(world.get_agenda(conservative=True)) == {'<List[Row],ComparableColumn:List[Row]> -> argmin',
                                                            '<List[Row]:List[Row]> -> next'}

        tokens = [Token(x) for x in ['what', 'was', 'the', 'attendance', 'before', 'the',
                                     'time', 'with', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],ComparableColumn:List[Row]> -> argmin',
                                           'StringColumn -> string_column:avg_attendance',
                                           '<List[Row]:List[Row]> -> previous',
                                           'NumberColumn -> number_column:avg_attendance'}
        # conservative disallows "before" mapping to "previous"
        assert set(world.get_agenda(conservative=True)) == {'<List[Row],ComparableColumn:List[Row]> -> argmin'}

        tokens = [Token(x) for x in ['what', 'was', 'the', 'attendance', 'above', 'the',
                                     'row', 'with', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],ComparableColumn:List[Row]> -> argmin',
                                           'StringColumn -> string_column:avg_attendance',
                                           '<List[Row]:List[Row]> -> previous',
                                           'NumberColumn -> number_column:avg_attendance'}
        assert set(world.get_agenda(conservative=True)) == {'<List[Row],ComparableColumn:List[Row]> -> argmin',
                                                            '<List[Row]:List[Row]> -> previous'}

        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'same', 'as', 'when',
                                     'the', 'league', 'was', 'usl', 'a', 'league', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'StringColumn -> string_column:avg_attendance',
                                           'NumberColumn -> number_column:avg_attendance',
                                           'StringColumn -> string_column:league',
                                           'List[str] -> string:usl_a_league',
                                           '<List[Row],Column:List[Row]> -> same_as',
                                           '<List[Row],DateColumn:Date> -> select_date'}
        assert set(world.get_agenda(conservative=True)) == {'StringColumn -> string_column:league',
                                                            'List[str] -> string:usl_a_league',
                                                            '<List[Row],Column:List[Row]> -> same_as',
                                                            '<List[Row],DateColumn:Date> -> select_date'}

        tokens = [Token(x) for x in ['what', 'is', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],NumberColumn:Number> -> min_number',
                                           'StringColumn -> string_column:avg_attendance',
                                           'NumberColumn -> number_column:avg_attendance'}
        assert set(world.get_agenda(conservative=True)) == {'<List[Row],NumberColumn:Number> -> min_number'}

        tokens = [Token(x) for x in ['when', 'did', 'the', 'team', 'not', 'qualify', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],DateColumn:Date> -> select_date',
                                           'List[str] -> string:qualify'}
        assert set(world.get_agenda(conservative=True)) == {'<List[Row],DateColumn:Date> -> select_date',
                                                            'List[str] -> string:qualify'}

        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'at', 'least',
                                     '7000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {
                '<List[Row],NumberColumn,Number:List[Row]> -> filter_number_greater_equals',
                '<List[Row],DateColumn:Date> -> select_date',
                'NumberColumn -> number_column:avg_attendance',
                'StringColumn -> string_column:avg_attendance',
                'Number -> 7000'}
        assert set(world.get_agenda(conservative=True)) == {
                '<List[Row],NumberColumn,Number:List[Row]> -> filter_number_greater_equals',
                '<List[Row],DateColumn:Date> -> select_date',
                'Number -> 7000'}

        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'more', 'than',
                                     '7000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],NumberColumn,Number:List[Row]> -> filter_number_greater',
                                           '<List[Row],DateColumn:Date> -> select_date',
                                           'NumberColumn -> number_column:avg_attendance',
                                           'StringColumn -> string_column:avg_attendance', 'Number -> 7000'}
        assert set(world.get_agenda(conservative=True)) == {
                '<List[Row],NumberColumn,Number:List[Row]> -> filter_number_greater',
                '<List[Row],DateColumn:Date> -> select_date',
                'Number -> 7000'}

        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'at', 'most',
                                     '7000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {
                '<List[Row],NumberColumn,Number:List[Row]> -> filter_number_lesser_equals',
                '<List[Row],DateColumn:Date> -> select_date',
                'NumberColumn -> number_column:avg_attendance',
                'StringColumn -> string_column:avg_attendance',
                'Number -> 7000'}
        assert set(world.get_agenda(conservative=True)) == {
                '<List[Row],NumberColumn,Number:List[Row]> -> filter_number_lesser_equals',
                '<List[Row],DateColumn:Date> -> select_date',
                'Number -> 7000'}

        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'no', 'more',
                                     'than', '7000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {
                '<List[Row],NumberColumn,Number:List[Row]> -> filter_number_lesser_equals',
                '<List[Row],DateColumn:Date> -> select_date',
                'NumberColumn -> number_column:avg_attendance',
                'StringColumn -> string_column:avg_attendance',
                'Number -> 7000'}
        assert set(world.get_agenda(conservative=True)) == {
                '<List[Row],NumberColumn,Number:List[Row]> -> filter_number_lesser_equals',
                '<List[Row],DateColumn:Date> -> select_date',
                'Number -> 7000'}

        tokens = [Token(x) for x in ['what', 'was', 'the', 'top', 'year', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row]:List[Row]> -> first', 'StringColumn -> string_column:year',
                                           'NumberColumn -> number_column:year',
                                           'DateColumn -> date_column:year'}
        assert set(world.get_agenda(conservative=True)) == {'<List[Row]:List[Row]> -> first'}

        tokens = [Token(x) for x in ['what', 'was', 'the', 'year', 'in', 'the', 'bottom', 'row',
                                     '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row]:List[Row]> -> last', 'StringColumn -> string_column:year',
                                           'NumberColumn -> number_column:year',
                                           'DateColumn -> date_column:year'}
        assert set(world.get_agenda(conservative=True)) == {'<List[Row]:List[Row]> -> last'}
