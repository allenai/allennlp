# pylint: disable=no-self-use,invalid-name,too-many-public-methods
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import WordTokenizer
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.domain_languages.domain_language import ExecutionError
from allennlp.semparse.domain_languages.wikitables_language import Date, WikiTablesLanguage


class TestWikiTablesLanguage(AllenNlpTestCase):
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
        assert cell_list == 'usl_a_league'

    def test_execute_works_with_argmax_on_dates(self):
        logical_form = "(select (argmax all_rows date_column:year) string_column:league)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == 'usl_first_division'

    def test_execute_works_with_argmin(self):
        logical_form = "(select (argmin all_rows number_column:avg_attendance) date_column:year)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == '2005'

    def test_execute_works_with_argmin_on_dates(self):
        logical_form = "(select (argmin all_rows date_column:year) string_column:league)"
        cell_list = self.language.execute(logical_form)
        assert cell_list == 'usl_a_league'

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
        assert self.language.execute(logical_form) == '4th_western'

    def test_execute_returns_none_with_first_on_empty_list(self):
        # Selecting "regular season" from the first row where year is greater than 2010.
        logical_form = """(select (first (filter_date_greater all_rows date_column:year
                                            (date 2010 -1 -1)))
                                  string_column:regular_season)"""
        assert self.language.execute(logical_form) is None

    def test_execute_works_with_last(self):
        # Selecting "regular season" from the last row where year is not equal to 2010.
        logical_form = """(select (last (filter_date_not_equals all_rows date_column:year
                                         (date 2010 -1 -1)))
                                  string_column:regular_season)"""
        assert self.language.execute(logical_form) == '5th'

    def test_execute_returns_none_with_last_on_empty_list(self):
        # Selecting "regular season" from the last row where year is greater than 2010.
        logical_form = """(select (last (filter_date_greater all_rows date_column:year (date 2010 -1 -1)))
                                      string_column:regular_season)"""
        assert self.language.execute(logical_form) is None

    def test_execute_works_with_previous(self):
        # Selecting "venue" from the row before last where year is not equal to 2010.
        logical_form = """(select (previous (last (filter_date_not_equals
                                                    all_rows date_column:year (date 2010 -1 -1))))
                                  string_column:regular_season)"""
        assert self.language.execute(logical_form) == "4th_western"

    def test_execute_returns_none_with_previous_on_empty_list(self):
        # Selecting "regular season" from the row before the one where year is greater than 2010.
        logical_form = """(select (previous (first (filter_date_greater all_rows date_column:year (date 2010 -1 -1))))
                                      string_column:regular_season)"""
        assert self.language.execute(logical_form) is None

    def test_execute_works_with_next(self):
        # Selecting "regular season" from the row after first where year is not equal to 2010.
        logical_form = """(select (next (first (filter_date_not_equals
                                                all_rows date_column:year (date 2010 -1 -1))))
                                  string_column:regular_season)"""
        assert self.language.execute(logical_form) == '5th'

    def test_execute_returns_none_with_next_on_empty_list(self):
        # Selecting "regular season" from the row after the one where year is greater than 2010.
        logical_form = """(select (next (first (filter_date_greater all_rows date_column:year (date 2010 -1 -1))))
                                      string_column:regular_season)"""
        assert self.language.execute(logical_form) is None

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
        assert result == "november_10"

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
