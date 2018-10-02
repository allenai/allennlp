# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.world import ExecutionError
from allennlp.semparse.executors import WikiTablesVariableFreeExecutor
from allennlp.semparse.executors.wikitables_variable_free_executor import Date


class TestWikiTablesVariableFreeExecutor(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        table_data = [{"fb:row.row.year": "fb:cell.2001", "fb:row.row.division": "fb:cell.2",
                       "fb:row.row.league": "fb:cell.usl_a_league", "fb:row.row.regular_season":
                       "fb:cell.4th_western", "fb:row.row.playoffs": "fb:cell.quarterfinals",
                       "fb:row.row.open_cup": "fb:cell.did_not_qualify",
                       "fb:row.row.avg_attendance": "fb:cell.7169"},
                      {"fb:row.row.year": "fb:cell.2005", "fb:row.row.division": "fb:cell.2",
                       "fb:row.row.league": "fb:cell.usl_first_division", "fb:row.row.regular_season":
                       "fb:cell.5th", "fb:row.row.playoffs": "fb:cell.quarterfinals",
                       "fb:row.row.open_cup": "fb:cell.4th_round",
                       "fb:row.row.avg_attendance": "fb:cell.6028"}]
        self.executor = WikiTablesVariableFreeExecutor(table_data)
        table_data_with_date = [{"fb:row.row.date": "fb:cell.january_2001", "fb:row.row.division": "fb:cell.2",
                                 "fb:row.row.league": "fb:cell.usl_a_league", "fb:row.row.regular_season":
                                 "fb:cell.4th_western", "fb:row.row.playoffs": "fb:cell.quarterfinals",
                                 "fb:row.row.open_cup": "fb:cell.did_not_qualify",
                                 "fb:row.row.avg_attendance": "fb:cell.7169"},
                                {"fb:row.row.date": "fb:cell.march_2005", "fb:row.row.division": "fb:cell.2",
                                 "fb:row.row.league": "fb:cell.usl_first_division", "fb:row.row.regular_season":
                                 "fb:cell.5th", "fb:row.row.playoffs": "fb:cell.quarterfinals",
                                 "fb:row.row.open_cup": "fb:cell.4th_round",
                                 "fb:row.row.avg_attendance": "fb:cell.6028"}]
        self.executor_with_date = WikiTablesVariableFreeExecutor(table_data_with_date)

    def test_execute_fails_with_unknown_function(self):
        logical_form = "(unknown_function all_rows fb:row.row.league)"
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_fails_with_unknown_constant(self):
        logical_form = "12fdw"
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_select(self):
        logical_form = "(select all_rows fb:row.row.league)"
        cell_list = self.executor.execute(logical_form)
        assert set(cell_list) == {'fb:cell.usl_a_league', 'fb:cell.usl_first_division'}

    def test_execute_works_with_argmax(self):
        logical_form = "(select (argmax all_rows fb:row.row.avg_attendance) fb:row.row.league)"
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ['fb:cell.usl_a_league']

    def test_execute_works_with_argmax_on_dates(self):
        logical_form = "(select (argmax all_rows fb:row.row.date) fb:row.row.league)"
        cell_list = self.executor_with_date.execute(logical_form)
        assert cell_list == ['fb:cell.usl_first_division']

    def test_execute_works_with_argmin(self):
        logical_form = "(select (argmin all_rows fb:row.row.avg_attendance) fb:row.row.year)"
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ['fb:cell.2005']

    def test_execute_works_with_argmin_on_dates(self):
        logical_form = "(select (argmin all_rows fb:row.row.date) fb:row.row.league)"
        cell_list = self.executor_with_date.execute(logical_form)
        assert cell_list == ['fb:cell.usl_a_league']

    def test_execute_works_with_filter_number_greater(self):
        # Selecting cell values from all rows that have attendance greater than the min value of
        # attendance.
        logical_form = """(select (filter_number_greater all_rows fb:row.row.avg_attendance
                                   (min all_rows fb:row.row.avg_attendance)) fb:row.row.league)"""
        cell_value_list = self.executor.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_number_greater all_rows fb:row.row.avg_attendance
                                   all_rows) fb:row.row.league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_greater(self):
        # Selecting cell values from all rows that have date greater than 2002.
        logical_form = """(select (filter_date_greater all_rows fb:row.row.date
                                   (date 2002 -1 -1)) fb:row.row.league)"""
        cell_value_list = self.executor_with_date.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_greater all_rows fb:row.row.date
                                   2005) fb:row.row.league)"""
        with self.assertRaises(ExecutionError):
            self.executor_with_date.execute(logical_form)

    def test_execute_works_with_filter_number_greater_equals(self):
        # Counting rows that have attendance greater than or equal to the min value of attendance.
        logical_form = """(count (filter_number_greater_equals all_rows fb:row.row.avg_attendance
                                  (min all_rows fb:row.row.avg_attendance)))"""
        count_result = self.executor.execute(logical_form)
        assert count_result == 2
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_greater all_rows fb:row.row.avg_attendance all_rows))"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_greater_equals(self):
        # Selecting cell values from all rows that have date greater than or equal to 2005 February
        # 1st.
        logical_form = """(select (filter_date_greater_equals all_rows fb:row.row.date
                                   (date 2005 2 1)) fb:row.row.league)"""
        cell_value_list = self.executor_with_date.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_greater_equals all_rows fb:row.row.date
                                   2005) fb:row.row.league)"""
        with self.assertRaises(ExecutionError):
            self.executor_with_date.execute(logical_form)

    def test_execute_works_with_filter_number_lesser(self):
        # Selecting cell values from all rows that have date lesser than 2005.
        logical_form = """(select (filter_number_lesser all_rows fb:row.row.year
                                    (max all_rows fb:row.row.year)) fb:row.row.league)"""
        cell_value_list = self.executor.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_number_lesser all_rows fb:row.row.year
                                   (date 2005 -1 -1)) fb:row.row.league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_lesser(self):
        # Selecting cell values from all rows that have date less that 2005 January
        logical_form = """(select (filter_date_lesser all_rows fb:row.row.date
                                   (date 2005 1 -1)) fb:row.row.league)"""
        cell_value_list = self.executor_with_date.execute(logical_form)
        assert cell_value_list == ["fb:cell.usl_a_league"]
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_lesser all_rows fb:row.row.date
                                   2005) fb:row.row.league)"""
        with self.assertRaises(ExecutionError):
            self.executor_with_date.execute(logical_form)

    def test_execute_works_with_filter_number_lesser_equals(self):
        # Counting rows that have year lesser than or equal to 2005.
        logical_form = """(count (filter_number_lesser_equals all_rows fb:row.row.year 2005))"""
        count_result = self.executor.execute(logical_form)
        assert count_result == 2
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_number_lesser_equals all_rows fb:row.row.year
                                   (date 2005 -1 -1)) fb:row.row.league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_lesser_equals(self):
        # Selecting cell values from all rows that have date less that or equal to 2001 February 23
        logical_form = """(select (filter_date_lesser_equals all_rows fb:row.row.date
                                   (date 2001 2 23)) fb:row.row.league)"""
        cell_value_list = self.executor_with_date.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_lesser_equals all_rows fb:row.row.date
                                   2005) fb:row.row.league)"""
        with self.assertRaises(ExecutionError):
            self.executor_with_date.execute(logical_form)

    def test_execute_works_with_filter_number_equals(self):
        # Counting rows that have year equal to 2010.
        logical_form = """(count (filter_number_equals all_rows fb:row.row.year 2010))"""
        count_result = self.executor.execute(logical_form)
        assert count_result == 0
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_equals all_rows fb:row.row.year (date 2010 -1 -1))"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_equals(self):
        # Selecting cell values from all rows that have date not equal to 2001
        logical_form = """(select (filter_date_equals all_rows fb:row.row.date
                                   (date 2001 -1 -1)) fb:row.row.league)"""
        cell_value_list = self.executor_with_date.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_equals all_rows fb:row.row.date
                                   2005) fb:row.row.league)"""
        with self.assertRaises(ExecutionError):
            self.executor_with_date.execute(logical_form)

    def test_execute_works_with_filter_number_not_equals(self):
        # Counting rows that have year not equal to 2010.
        logical_form = """(count (filter_number_not_equals all_rows fb:row.row.year 2010))"""
        count_result = self.executor.execute(logical_form)
        assert count_result == 2
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_not_equals all_rows fb:row.row.year (date 2010 -1 -1))"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_not_equals(self):
        # Selecting cell values from all rows that have date not equal to 2001
        logical_form = """(select (filter_date_not_equals all_rows fb:row.row.date
                                   (date 2001 -1 -1)) fb:row.row.league)"""
        cell_value_list = self.executor_with_date.execute(logical_form)
        assert cell_value_list == ['fb:cell.usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select (filter_date_not_equals all_rows fb:row.row.date
                                   2005) fb:row.row.league)"""
        with self.assertRaises(ExecutionError):
            self.executor_with_date.execute(logical_form)

    def test_execute_works_with_filter_in(self):
        # Selecting "regular season" from rows that have "did not qualify" in "open cup" column.
        logical_form = """(select (filter_in all_rows fb:row.row.open_cup did_not_qualify)
                                  fb:row.row.regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["fb:cell.4th_western"]

    def test_execute_works_with_filter_not_in(self):
        # Selecting "regular season" from rows that do not have "did not qualify" in "open cup" column.
        logical_form = """(select (filter_not_in all_rows fb:row.row.open_cup did_not_qualify)
                                   fb:row.row.regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["fb:cell.5th"]

    def test_execute_works_with_first(self):
        # Selecting "regular season" from the first row.
        logical_form = """(select (first all_rows) fb:row.row.regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["fb:cell.4th_western"]

    def test_execute_logs_warning_with_first_on_empty_list(self):
        # Selecting "regular season" from the first row where year is greater than 2010.
        with self.assertLogs("allennlp.semparse.executors.wikitables_variable_free_executor") as log:
            logical_form = """(select (first (filter_number_greater all_rows fb:row.row.year 2010))
                                      fb:row.row.regular_season)"""
            self.executor.execute(logical_form)
        self.assertEqual(log.output,
                         ["WARNING:allennlp.semparse.executors.wikitables_variable_free_executor:"
                          "Trying to get first row from an empty list: "
                          "['filter_number_greater', 'all_rows', 'fb:row.row.year', '2010']"])

    def test_execute_works_with_last(self):
        # Selecting "regular season" from the last row where year is not equal to 2010.
        logical_form = """(select (last (filter_number_not_equals all_rows fb:row.row.year 2010))
                                  fb:row.row.regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["fb:cell.5th"]

    def test_execute_logs_warning_with_last_on_empty_list(self):
        # Selecting "regular season" from the last row where year is greater than 2010.
        with self.assertLogs("allennlp.semparse.executors.wikitables_variable_free_executor") as log:
            logical_form = """(select (last (filter_number_greater all_rows fb:row.row.year 2010))
                                      fb:row.row.regular_season)"""
            self.executor.execute(logical_form)
        self.assertEqual(log.output,
                         ["WARNING:allennlp.semparse.executors.wikitables_variable_free_executor:"
                          "Trying to get last row from an empty list: "
                          "['filter_number_greater', 'all_rows', 'fb:row.row.year', '2010']"])

    def test_execute_works_with_previous(self):
        # Selecting "regular season" from the row before last where year is not equal to 2010.
        logical_form = """(select (previous (last (filter_number_not_equals
                                                    all_rows fb:row.row.year 2010)))
                                  fb:row.row.regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["fb:cell.4th_western"]

    def test_execute_logs_warning_with_previous_on_empty_list(self):
        # Selecting "regular season" from the row before the one where year is greater than 2010.
        with self.assertLogs("allennlp.semparse.executors.wikitables_variable_free_executor") as log:
            logical_form = """(select (previous (filter_number_greater all_rows fb:row.row.year 2010))
                                      fb:row.row.regular_season)"""
            self.executor.execute(logical_form)
        self.assertEqual(log.output,
                         ["WARNING:allennlp.semparse.executors.wikitables_variable_free_executor:"
                          "Trying to get the previous row from an empty list: "
                          "['filter_number_greater', 'all_rows', 'fb:row.row.year', '2010']"])

    def test_execute_works_with_next(self):
        # Selecting "regular season" from the row after first where year is not equal to 2010.
        logical_form = """(select (next (first (filter_number_not_equals
                                                    all_rows fb:row.row.year 2010)))
                                  fb:row.row.regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["fb:cell.5th"]

    def test_execute_logs_warning_with_next_on_empty_list(self):
        # Selecting "regular season" from the row after the one where year is greater than 2010.
        with self.assertLogs("allennlp.semparse.executors.wikitables_variable_free_executor") as log:
            logical_form = """(select (next (filter_number_greater all_rows fb:row.row.year 2010))
                                      fb:row.row.regular_season)"""
            self.executor.execute(logical_form)
        self.assertEqual(log.output,
                         ["WARNING:allennlp.semparse.executors.wikitables_variable_free_executor:"
                          "Trying to get the next row from an empty list: "
                          "['filter_number_greater', 'all_rows', 'fb:row.row.year', '2010']"])

    def test_execute_works_with_mode(self):
        # Most frequent division value.
        logical_form = """(mode all_rows fb:row.row.division)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["fb:cell.2"]
        # If we used select instead, we should get a list of two values.
        logical_form = """(select all_rows fb:row.row.division)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["fb:cell.2", "fb:cell.2"]
        # If we queried for the most frequent year instead, it should return two values since both
        # have the max frequency of 1.
        logical_form = """(mode all_rows fb:row.row.year)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["fb:cell.2001", "fb:cell.2005"]

    def test_execute_works_with_same_as(self):
        # Select the "league" from all the rows that have the same value under "playoffs" as the
        # row that has the string "a league" under "league".
        logical_form = """(select (same_as (filter_in all_rows fb:row.row.league a_league)
                                   fb:row.row.playoffs)
                           fb:row.row.league)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["fb:cell.usl_a_league", "fb:cell.usl_first_division"]

    def test_execute_works_with_sum(self):
        # Get total "avg attendance".
        logical_form = """(sum all_rows fb:row.row.avg_attendance)"""
        sum_value = self.executor.execute(logical_form)
        assert sum_value == 13197
        # Total "avg attendance" where "playoffs" has "quarterfinals"
        logical_form = """(sum (filter_in all_rows fb:row.row.playoffs quarterfinals)
                                fb:row.row.avg_attendance)"""
        sum_value = self.executor.execute(logical_form)
        assert sum_value == 13197

    def test_execute_works_with_average(self):
        # Get average "avg attendance".
        logical_form = """(average all_rows fb:row.row.avg_attendance)"""
        avg_value = self.executor.execute(logical_form)
        assert avg_value == 6598.5
        # Average "avg attendance" where "playoffs" has "quarterfinals"
        logical_form = """(average (filter_in all_rows fb:row.row.playoffs quarterfinals)
                                fb:row.row.avg_attendance)"""
        avg_value = self.executor.execute(logical_form)
        assert avg_value == 6598.5

    def test_execute_works_with_diff(self):
        # Difference in "avg attendance" between rows with "usl_a_league" and "usl_first_division"
        # in "league" columns.
        logical_form = """(diff (filter_in all_rows fb:row.row.league usl_a_league)
                                (filter_in all_rows fb:row.row.league usl_first_division)
                                fb:row.row.avg_attendance)"""
        avg_value = self.executor.execute(logical_form)
        assert avg_value == 1141

    def test_execute_fails_with_diff_on_non_numerical_columns(self):
        logical_form = """(diff (filter_in all_rows fb:row.row.league usl_a_league)
                                (filter_in all_rows fb:row.row.league usl_first_division)
                                fb:row.row.league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_fails_with_non_int_dates(self):
        logical_form = """(date 2015 1.5 1)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_date_comparison_works(self):
        assert Date(2013, 12, 31) > Date(2013, 12, 30)
        assert Date(2013, 12, 31) == Date(2013, 12, -1)
        assert Date(2013, -1, -1) >= Date(2013, 12, 31)
        # pylint: disable=singleton-comparison
        assert (Date(2013, 12, -1) > Date(2013, 12, 31)) == False
        assert (Date(2013, 12, 31) > 2013) == False
        assert (Date(2013, 12, 31) >= 2013) == False
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
