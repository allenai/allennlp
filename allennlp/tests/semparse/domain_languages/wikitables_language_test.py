# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from typing import List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import WordTokenizer
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.domain_languages.domain_language import ExecutionError
from allennlp.semparse.domain_languages.wikitables_language import Date, WikiTablesLanguage
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

    def test_execute_fails_with_unknown_function(self):
        logical_form = "(unknown_function all_rows string_column:league)"
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_select(self):
        logical_form = "(select all_rows string_column:league)"
        cell_list = self.language.execute(logical_form)
        assert set(cell_list) == {'usl_a_league', 'usl_first_division'}

    def test_execute_works_with_argmax(self):
        logical_form = "(select (argmax all_rows number_column:avg_attendance) string_column:league)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == ['usl_a_league']

    def test_execute_works_with_argmax_on_dates(self):
        logical_form = "(select (argmax all_rows date_column:year) string_column:league)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == ['usl_first_division']

    def test_execute_works_with_argmin(self):
        logical_form = "(select (argmin all_rows number_column:avg_attendance) date_column:year)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == ['2005']

    def test_execute_works_with_argmin_on_dates(self):
        logical_form = "(select (argmin all_rows date_column:year) string_column:league)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == ['usl_a_league']

    def test_execute_works_with_filter_number_greater(self):
        # Selecting cell values from all rows that have attendance greater than the min value of
        # attendance.
        logical_form = """(select (filter_number_greater all_rows number_column:avg_attendance
                                   (min all_rows number_column:avg_attendance)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_number_greater all_rows number_column:avg_attendance
                                   all_rows) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_date_greater(self):
        # Selecting cell values from all rows that have date greater than 2002.
        logical_form = """(select (filter_date_greater all_rows date_column:year
                                   (date 2002 -1 -1)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_greater all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_number_greater_equals(self):
        # Counting rows that have attendance greater than or equal to the min value of attendance.
        logical_form = """(count (filter_number_greater_equals all_rows number_column:avg_attendance
                                  (min all_rows number_column:avg_attendance)))"""
        count_result = self.language.execute(logical_form)
        assert count_result == 2
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_greater all_rows number_column:avg_attendance all_rows))"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_date_greater_equals(self):
        # Selecting cell values from all rows that have date greater than or equal to 2005 February
        # 1st.
        logical_form = """(select (filter_date_greater_equals all_rows date_column:year
                                   (date 2005 2 1)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_greater_equals all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_number_lesser(self):
        # Selecting cell values from all rows that have date lesser than 2005.
        logical_form = """(select (filter_number_lesser all_rows number_column:avg_attendance
                                    (max all_rows number_column:avg_attendance)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_number_lesser all_rows date_column:year
                                   (date 2005 -1 -1)) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_date_lesser(self):
        # Selecting cell values from all rows that have date less that 2005 January
        logical_form = """(select (filter_date_lesser all_rows date_column:year
                                   (date 2005 1 -1)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ["usl_a_league"]
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_lesser all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_number_lesser_equals(self):
        # Counting rows that have year lesser than or equal to 2005.
        logical_form = """(count (filter_number_lesser_equals all_rows number_column:avg_attendance 8000))"""
        count_result = self.language.execute(logical_form)
        assert count_result == 2
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_number_lesser_equals all_rows date_column:year
                                   (date 2005 -1 -1)) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_date_lesser_equals(self):
        # Selecting cell values from all rows that have date less that or equal to 2001 February 23
        logical_form = """(select (filter_date_lesser_equals all_rows date_column:year
                                   (date 2001 2 23)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_lesser_equals all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_number_equals(self):
        # Counting rows that have year equal to 2010.
        logical_form = """(count (filter_number_equals all_rows number_column:avg_attendance 8000))"""
        count_result = self.language.execute(logical_form)
        assert count_result == 0
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_equals all_rows date_column:year (date 2010 -1 -1)))"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_date_equals(self):
        # Selecting cell values from all rows that have date not equal to 2001
        logical_form = """(select (filter_date_equals all_rows date_column:year
                                   (date 2001 -1 -1)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_equals all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_number_not_equals(self):
        # Counting rows that have average attendance not equal to 8000
        logical_form = """(count (filter_number_not_equals all_rows number_column:avg_attendance 8000))"""
        count_result = self.language.execute(logical_form)
        assert count_result == 2
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_not_equals all_rows date_column:year (date 2010 -1 -1)))"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_date_not_equals(self):
        # Selecting cell values from all rows that have date not equal to 2001
        logical_form = """(select (filter_date_not_equals all_rows date_column:year
                                   (date 2001 -1 -1)) string_column:league)"""
        cell_value_list = self.language.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_not_equals all_rows date_column:year
                                   2005) string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_filter_in(self):
        # Selecting "regular season" from rows that have "did not qualify" in "open cup" column.
        logical_form = """(select (filter_in all_rows string_column:open_cup string:did_not_qualify)
                                  string_column:regular_season)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["4th_western"]

    def test_execute_works_with_filter_not_in(self):
        # Selecting "regular season" from rows that do not have "did not qualify" in "open cup" column.
        logical_form = """(select (filter_not_in all_rows string_column:open_cup string:did_not_qualify)
                                   string_column:regular_season)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["5th"]

    def test_execute_works_with_first(self):
        # Selecting "regular season" from the first row.
        logical_form = """(select (first all_rows) string_column:regular_season)"""
        assert self.language.execute(logical_form) == ['4th_western']

    def test_execute_returns_empty_list_with_first_on_empty_list(self):
        # Selecting "regular season" from the first row where year is greater than 2010.
        logical_form = """(select (first (filter_date_greater all_rows date_column:year
                                            (date 2010 -1 -1)))
                                  string_column:regular_season)"""
        assert self.language.execute(logical_form) == []

    def test_execute_works_with_last(self):
        # Selecting "regular season" from the last row where year is not equal to 2010.
        logical_form = """(select (last (filter_date_not_equals all_rows date_column:year
                                         (date 2010 -1 -1)))
                                  string_column:regular_season)"""
        assert self.language.execute(logical_form) == ['5th']

    def test_execute_returns_empty_list_with_last_on_empty_list(self):
        # Selecting "regular season" from the last row where year is greater than 2010.
        logical_form = """(select (last (filter_date_greater all_rows date_column:year (date 2010 -1 -1)))
                                      string_column:regular_season)"""
        assert self.language.execute(logical_form) == []

    def test_execute_works_with_previous(self):
        # Selecting "venue" from the row before last where year is not equal to 2010.
        logical_form = """(select (previous (last (filter_date_not_equals
                                                    all_rows date_column:year (date 2010 -1 -1))))
                                  string_column:regular_season)"""
        assert self.language.execute(logical_form) == ["4th_western"]

    def test_execute_returns_empty_list_with_previous_on_empty_list(self):
        # Selecting "regular season" from the row before the one where year is greater than 2010.
        logical_form = """(select (previous (first (filter_date_greater all_rows date_column:year
                                                                        (date 2010 -1 -1))))
                                  string_column:regular_season)"""
        assert self.language.execute(logical_form) == []

    def test_execute_works_with_next(self):
        # Selecting "regular season" from the row after first where year is not equal to 2010.
        logical_form = """(select (next (first (filter_date_not_equals
                                                all_rows date_column:year (date 2010 -1 -1))))
                                  string_column:regular_season)"""
        assert self.language.execute(logical_form) == ['5th']

    def test_execute_returns_empty_list_with_next_on_empty_list(self):
        # Selecting "regular season" from the row after the one where year is greater than 2010.
        logical_form = """(select (next (first (filter_date_greater all_rows date_column:year (date 2010 -1 -1))))
                                      string_column:regular_season)"""
        assert self.language.execute(logical_form) == []

    def test_execute_works_with_mode(self):
        # Most frequent division value.
        logical_form = """(mode all_rows number_column:division)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["2"]
        # If we used select instead, we should get a list of two values.
        logical_form = """(select all_rows number_column:division)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["2", "2"]
        # If we queried for the most frequent year instead, it should return two values since both
        # have the max frequency of 1.
        logical_form = """(mode all_rows date_column:year)"""
        cell_list = self.language.execute(logical_form)
        assert cell_list == ["2001", "2005"]

    def test_execute_works_with_same_as(self):
        # Select the "league" from all the rows that have the same value under "playoffs" as the
        # row that has the string "a league" under "league".
        logical_form = """(select (same_as (first (filter_in all_rows string_column:league string:a_league))
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
        logical_form = """(diff (first (filter_in all_rows string_column:league string:usl_a_league))
                                (first (filter_in all_rows string_column:league string:usl_first_division))
                                number_column:avg_attendance)"""
        avg_value = self.language.execute(logical_form)
        assert avg_value == 1141

    def test_execute_fails_with_diff_on_non_numerical_columns(self):
        logical_form = """(diff (first (filter_in all_rows string_column:league string:usl_a_league))
                                (first (filter_in all_rows string_column:league string:usl_first_division))
                                string_column:league)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_fails_with_non_int_dates(self):
        logical_form = """(date 2015 1.5 1)"""
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_date_comparison_works(self):
        assert Date(2013, 12, 31) > Date(2013, 12, 30)
        assert Date(2013, 12, 31) == Date(2013, 12, -1)
        assert Date(2013, -1, -1) >= Date(2013, 12, 31)
        # pylint: disable=singleton-comparison
        assert (Date(2013, 12, -1) > Date(2013, 12, 31)) == False
        with pytest.raises(ExecutionError, match='only compare Dates with Dates'):
            assert (Date(2013, 12, 31) > 2013) == False
        with pytest.raises(ExecutionError, match='only compare Dates with Dates'):
            assert (Date(2013, 12, 31) >= 2013) == False
        with pytest.raises(ExecutionError, match='only compare Dates with Dates'):
            assert Date(2013, 12, 31) != 2013
        assert (Date(2018, 1, 1) >= Date(-1, 2, 1)) == False
        assert (Date(2018, 1, 1) < Date(-1, 2, 1)) == False
        # When year is unknown in both cases, we can compare months and days.
        assert Date(-1, 2, 1) < Date(-1, 2, 3)
        # If both year and month are not know in both cases, the comparison is undefined, and both
        # < and >= return False.
        assert (Date(-1, -1, 1) < Date(-1, -1, 3)) == False
        assert (Date(-1, -1, 1) >= Date(-1, -1, 3)) == False
        # Same when year is known, buth months are not.
        assert (Date(2018, -1, 1) < Date(2018, -1, 3)) == False
        # TODO (pradeep): Figure out whether this is expected behavior by looking at data.
        assert (Date(2018, -1, 1) >= Date(2018, -1, 3)) == False

    def test_number_comparison_works(self):
        # TableQuestionContext normlaizes all strings according to some rules. We want to ensure
        # that the original numerical values of number cells is being correctly processed here.
        tokens = WordTokenizer().tokenize("when was the attendance the highest?")
        tagged_file = self.FIXTURES_ROOT / "data" / "corenlp_processed_tables" / "TEST-2.table"
        context = TableQuestionContext.read_from_file(tagged_file, tokens)
        language = WikiTablesLanguage(context)
        result = language.execute("(select (argmax all_rows number_column:attendance) date_column:date)")
        assert result == ["november_10"]

    def test_evaluate_logical_form(self):
        logical_form = """(select (same_as (first (filter_in all_rows string_column:league string:a_league))
                                   string_column:playoffs)
                           string_column:league)"""
        assert self.language.evaluate_logical_form(logical_form, ["USL A-League",
                                                                  "USL First Division"])

    def test_evaluate_logical_form_with_invalid_logical_form(self):
        logical_form = """(select (same_as (first (filter_in all_rows string_column:league INVALID_CONSTANT))
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
                "<List[Row],Column:List[str]>",
                "<List[Row],NumberColumn,Number:List[Row]>",
                "<List[Row],ComparableColumn:List[Row]>",
                "<List[Row],Column:List[Row]>",
                "<List[Row],List[Row],NumberColumn:Number>",
                "<List[Row],StringColumn,str:List[Row]>",
                "<Number,Number,Number:Date>",
                "<List[Row],DateColumn,Date:List[Row]>",
                "<List[Row],NumberColumn:Number>",
                "<List[Row]:List[Row]>",
                "<List[Row]:Number>",
                "List[str]",
                "List[Row]",
                "Date",
                "Number",
                "Column",
                "StringColumn",
                "ComparableColumn",
                "NumberColumn",
                "DateColumn",
                "str",
                }

        check_productions_match(productions['@start@'],
                                ['Date', 'Number', 'List[str]'])

        check_productions_match(productions['<List[Row],Column:List[str]>'],
                                ['mode', 'select'])

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

        check_productions_match(productions['<List[Row],StringColumn,str:List[Row]>'],
                                ['filter_in', 'filter_not_in'])

        check_productions_match(productions['<Number,Number,Number:Date>'],
                                ['date'])

        check_productions_match(productions['<List[Row],DateColumn,Date:List[Row]>'],
                                ['filter_date_equals', 'filter_date_greater',
                                 'filter_date_greater_equals', 'filter_date_lesser',
                                 'filter_date_lesser_equals', 'filter_date_not_equals'])

        check_productions_match(productions['<List[Row],NumberColumn:Number>'],
                                ['average', 'max', 'min', 'sum'])

        check_productions_match(productions['<List[Row]:List[Row]>'],
                                ['first', 'last', 'next', 'previous'])

        check_productions_match(productions['<List[Row]:Number>'],
                                ['count'])

        check_productions_match(productions['List[str]'],
                                ['[<List[Row],Column:List[str]>, List[Row], Column]'])

        check_productions_match(productions['List[Row]'],
                                ['all_rows',
                                 '[<List[Row],DateColumn,Date:List[Row]>, List[Row], DateColumn, Date]',
                                 '[<List[Row],Column:List[Row]>, List[Row], Column]',
                                 '[<List[Row],ComparableColumn:List[Row]>, List[Row], ComparableColumn]',
                                 '[<List[Row],NumberColumn,Number:List[Row]>, List[Row], NumberColumn, Number]',
                                 '[<List[Row],StringColumn,str:List[Row]>, List[Row], StringColumn, str]',
                                 '[<List[Row]:List[Row]>, List[Row]]'])

        check_productions_match(productions['Date'],
                                ['[<Number,Number,Number:Date>, Number, Number, Number]'])

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
        check_productions_match(productions['Column'],
                                ['string_column:league',
                                 'string_column:playoffs',
                                 'string_column:open_cup',
                                 'string_column:regular_season',
                                 'date_column:year',
                                 'number_column:avg_attendance',
                                 'number_column:division'])

        check_productions_match(productions['StringColumn'],
                                ['string_column:league',
                                 'string_column:playoffs',
                                 'string_column:open_cup',
                                 'string_column:regular_season'])

        check_productions_match(productions['ComparableColumn'],
                                ['date_column:year',
                                 'number_column:avg_attendance',
                                 'number_column:division'])

        check_productions_match(productions['DateColumn'],
                                ['date_column:year'])

        check_productions_match(productions['NumberColumn'],
                                ['number_column:avg_attendance',
                                 'number_column:division'])

        # Strings come from the question - any span in the question that shows up as a cell in the
        # table is a valid string production.
        check_productions_match(productions['str'],
                                ['string:quarterfinals',
                                 'string:did_not_qualify',
                                 'string:a_league',
                                 'string:usl_first_division',
                                 'string:usl_a_league'])

    def test_get_nonterminal_productions_in_world_without_number_columns(self):
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'first', 'title', '?']]
        table_file = self.FIXTURES_ROOT / 'data' / 'corenlp_processed_tables' / 'TEST-6.table'
        table_context = TableQuestionContext.read_from_file(table_file, question_tokens)
        # The table does not have a number column.
        assert "number" not in table_context.column_types.values()
        world = WikiTablesLanguage(table_context)
        actions = world.get_nonterminal_productions()
        assert set(actions.keys()) == {
                "<List[Row],Column:List[str]>",
                "<List[Row],ComparableColumn:List[Row]>",
                "<List[Row],Column:List[Row]>",
                "<List[Row],StringColumn,str:List[Row]>",
                "<Number,Number,Number:Date>",
                "<List[Row],DateColumn,Date:List[Row]>",
                "<List[Row]:List[Row]>",
                "<List[Row]:Number>",
                "Date",
                "Number",
                "List[str]",
                "Column",
                "ComparableColumn",
                "DateColumn",
                "StringColumn",
                "List[Row]",
                "@start@",
                }

    def test_get_nonterminal_productions_in_world_without_date_columns(self):
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'first', 'title', '?']]
        table_file = self.FIXTURES_ROOT / 'data' / 'corenlp_processed_tables' / 'TEST-4.table'
        table_context = TableQuestionContext.read_from_file(table_file, question_tokens)
        # The table does not have a date column.
        assert "date" not in table_context.column_types.values()
        world = WikiTablesLanguage(table_context)
        actions = world.get_nonterminal_productions()
        assert set(actions.keys()) == {
                "<List[Row],Column:List[str]>",
                "<List[Row],NumberColumn,Number:List[Row]>",
                "<List[Row],ComparableColumn:List[Row]>",
                "<List[Row],Column:List[Row]>",
                "<List[Row],List[Row],NumberColumn:Number>",
                "<List[Row],StringColumn,str:List[Row]>",
                "<Number,Number,Number:Date>",
                "<List[Row],NumberColumn:Number>",
                "<List[Row]:List[Row]>",
                "<List[Row]:Number>",
                "Date",
                "Number",
                "List[str]",
                "Column",
                "ComparableColumn",
                "StringColumn",
                "NumberColumn",
                "List[Row]",
                "@start@",
                }

    def test_get_nonterminal_productions_in_world_without_comparable_columns(self):
        question_tokens = [Token(x) for x in ['what', 'was', 'the', 'first', 'title', '?']]
        table_file = self.FIXTURES_ROOT / 'data' / 'corenlp_processed_tables' / 'TEST-1.table'
        table_context = TableQuestionContext.read_from_file(table_file, question_tokens)
        # The table does not have date or number columns.
        assert "date" not in table_context.column_types.values()
        assert "number" not in table_context.column_types.values()
        world = WikiTablesLanguage(table_context)
        actions = world.get_nonterminal_productions()
        assert set(actions.keys()) == {
                "<List[Row],Column:List[str]>",
                "<List[Row],Column:List[Row]>",
                "<List[Row],StringColumn,str:List[Row]>",
                "<Number,Number,Number:Date>",
                "<List[Row]:List[Row]>",
                "<List[Row]:Number>",
                "Date",
                "Number",
                "List[str]",
                "Column",
                "StringColumn",
                "List[Row]",
                "@start@",
                }

    def test_world_processes_logical_forms_correctly(self):
        logical_form = "(select (filter_in all_rows string_column:league string:usl_a_league) date_column:year)"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.action_sequence_to_logical_form(action_sequence) == logical_form

    def test_world_gets_correct_actions(self):
        logical_form = "(select (filter_in all_rows string_column:league string:usl_a_league) date_column:year)"
        expected_sequence = ['@start@ -> List[str]',
                             'List[str] -> [<List[Row],Column:List[str]>, List[Row], Column]',
                             '<List[Row],Column:List[str]> -> select',
                             'List[Row] -> [<List[Row],StringColumn,str:List[Row]>, '
                                     'List[Row], StringColumn, str]',  # pylint: disable=bad-continuation
                             '<List[Row],StringColumn,str:List[Row]> -> filter_in',
                             'List[Row] -> all_rows',
                             'StringColumn -> string_column:league',
                             'str -> string:usl_a_league',
                             'Column -> date_column:year']
        assert self.language.logical_form_to_action_sequence(logical_form) == expected_sequence

    def test_world_gets_logical_form_from_actions(self):
        logical_form = "(select (filter_in all_rows string_column:league string:usl_a_league) date_column:year)"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.action_sequence_to_logical_form(action_sequence) == logical_form

    def test_world_processes_logical_forms_with_number_correctly(self):
        logical_form = ("(select (filter_number_greater all_rows number_column:avg_attendance 8000) "
                        "date_column:year)")
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.action_sequence_to_logical_form(action_sequence) == logical_form

    def test_world_processes_logical_forms_with_date_correctly(self):
        logical_form = ("(select (filter_date_greater all_rows date_column:year (date 2013 -1 -1)) "
                        "date_column:year)")
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert self.language.action_sequence_to_logical_form(action_sequence) == logical_form

    def test_get_agenda(self):
        tokens = [Token(x) for x in ['what', 'was', 'the', 'last', 'year', '2000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'Number -> 2000',
                                           '<List[Row]:List[Row]> -> last',
                                           'DateColumn -> date_column:year'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'difference', 'in', 'attendance',
                                     'between', 'years', '2001', 'and', '2005', '?']]
        world = self._get_world_with_question_tokens(tokens)
        # "year" column does not match because "years" occurs in the question.
        assert set(world.get_agenda()) == {'Number -> 2001',
                                           'Number -> 2005',
                                           '<List[Row],List[Row],NumberColumn:Number> -> diff'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'total', 'avg.', 'attendance', 'in',
                                     'years', '2001', 'and', '2005', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'Number -> 2001',
                                           'Number -> 2005',
                                           '<List[Row],NumberColumn:Number> -> sum',
                                           'NumberColumn -> number_column:avg_attendance'}
        tokens = [Token(x) for x in ['when', 'was', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],ComparableColumn:List[Row]> -> argmin',
                                           'NumberColumn -> number_column:avg_attendance'}
        tokens = [Token(x) for x in ['what', 'is', 'the', 'least', 'avg.', 'attendance', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row],NumberColumn:Number> -> min',
                                           'NumberColumn -> number_column:avg_attendance'}
        tokens = [Token(x) for x in ['when', 'did', 'the', 'team', 'not', 'qualify', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'str -> string:qualify'}
        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'at', 'least',
                                     '7000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        agenda = set(world.get_agenda())
        assert agenda == {'<List[Row],NumberColumn,Number:List[Row]> -> filter_number_greater_equals',
                          'NumberColumn -> number_column:avg_attendance',
                          'Number -> 7000'}
        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'more', 'than', '7000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        agenda = set(world.get_agenda())
        assert agenda == {'<List[Row],NumberColumn,Number:List[Row]> -> filter_number_greater',
                          'NumberColumn -> number_column:avg_attendance',
                          'Number -> 7000'}
        tokens = [Token(x) for x in ['when', 'was', 'the', 'avg.', 'attendance', 'at', 'most', '7000', '?']]
        world = self._get_world_with_question_tokens(tokens)
        agenda = set(world.get_agenda())
        assert agenda == {'<List[Row],NumberColumn,Number:List[Row]> -> filter_number_lesser_equals',
                          'NumberColumn -> number_column:avg_attendance', 'Number -> 7000'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'top', 'year', '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row]:List[Row]> -> first', 'DateColumn -> date_column:year'}
        tokens = [Token(x) for x in ['what', 'was', 'the', 'year', 'in', 'the', 'bottom', 'row',
                                     '?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == {'<List[Row]:List[Row]> -> last', 'DateColumn -> date_column:year'}
