from typing import List, Dict, Tuple, Union
from collections import defaultdict
import logging

from overrides import overrides

from allennlp.semparse.worlds.world import ExecutionError
from allennlp.semparse.contexts.table_question_knowledge_graph import MONTH_NUMBERS
from allennlp.semparse.executors.executor import Executor, NestedList

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Date:
    def __init__(self, year: int, month: int, day: int) -> None:
        self.year = year
        self.month = month
        self.day = day

    def __eq__(self, other) -> bool:
        if not isinstance(other, Date):
            return False
        year_is_same = self.year == -1 or other.year == -1 or self.year == other.year
        month_is_same = self.month == -1 or other.month == -1 or self.month == other.month
        day_is_same = self.day == -1 or other.day == -1 or self.day == other.day
        return year_is_same and month_is_same and day_is_same

    def __gt__(self, other) -> bool:
        # pylint: disable=too-many-return-statements
        if not isinstance(other, Date):
            return False
        if self.year == -1 or other.year == -1:
            return False
        if self.year != other.year:
            return self.year > other.year
        # The years are equal and not -1
        if self.month == -1 or other.month == -1:
            return False
        if self.month != other.month:
            return self.month > other.month
        # The months and years are equal and not -1
        if self.day == -1 or other.day == -1:
            return False
        return self.day > other.day

    def __ge__(self, other) -> bool:
        if not isinstance(other, Date):
            return False
        return self > other or self == other



class WikiTablesVariableFreeExecutor(Executor):
    def __init__(self, table_data: List[Dict[str, str]]) -> None:
        super().__init__(table_data)
        self._table_data = table_data

    @overrides
    def _handle_expression(self, expression_list: NestedList):
        if isinstance(expression_list, list) and len(expression_list) == 1:
            expression = expression_list[0]
        else:
            expression = expression_list
        if isinstance(expression, list):
            # This is a function application.
            function_name = expression[0]
        else:
            # This is a constant (like "all_rows" or "2005")
            return self._handle_constant(str(expression))
        try:
            function = getattr(self, f"_{function_name}")
            return function(expression[1:])
        except AttributeError:
            logger.error("Function not found: %s", function_name)
            raise ExecutionError(f"Function not found: {function_name}")

    def _get_row_list_and_column_name(self, expression_list: NestedList) -> Tuple[List[Dict[str,
                                                                                            str]],
                                                                                  str]:
        """
        Utility function for computing the initial row list and a column name from an expression for
        all functions that need these operations, like "select", "argmax", "argmin", etc.
        """
        row_list: List[Dict[str, str]] = self._handle_expression(expression_list[0])
        column_name = expression_list[1]
        if not (isinstance(column_name, str) and column_name.startswith("fb:row.row.")):
            logger.error("Invalid column for selection: %s", column_name)
            raise ExecutionError(f"Invalid column for selection: {column_name}")
        if row_list and column_name not in row_list[0]:
            logger.error("Input list of rows do not contain column: %s", column_name)
            raise ExecutionError(f"Input list of rows do not contain column: {column_name}")
        return row_list, column_name

    def _handle_constant(self, constant: str) -> Union[List[Dict[str, str]], float]:
        if constant == "all_rows":
            return self._table_data
        try:
            return float(constant)
        except ValueError:
            logger.error(f"Cannot handle constant: {constant}")
            raise ExecutionError(f"Cannot handle constant: {constant}")

    def _select(self, expression_list: NestedList) -> List[str]:
        """
        Select function takes a list of rows and a column (decoded from the `expression_list`) and
        returns a list of cell values as strings.
        """
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        return [row[column_name] for row in row_list]

    def _get_number_row_pairs_to_filter(self, expression_list: NestedList) -> List[Tuple[float,
                                                                                         Dict[str, str]]]:
        """
        Helper method that takes an expression that evaluates to a row list and a column name, and
        returns a list of tuples, each containing as the first element a number taken from
        that column, and the corresponding row as the second element. The output can be used to
        compare rows based on the numbers.
        """
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        try:
            cell_row_pairs = [(float(row[column_name].replace('fb:cell.', '')), row) for row in row_list]
        except ValueError:
            # This means that at least one of the cells is not numerical.
            return []
        return cell_row_pairs

    def _get_date_row_pairs_to_filter(self, expression_list: NestedList) -> List[Tuple[Date,
                                                                                       Dict[str, str]]]:
        """
        Helper method that takes an expression that evaluates to a row list and a column name, and
        returns a list of tuples, each containing as the first element a date taken from
        that column, and the corresponding row as the second element. The output can be used to
        compare rows based on the dates.
        """
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        cell_row_pairs = [(self._make_date(row[column_name].replace('fb:cell.', '')), row)
                          for row in row_list]
        return cell_row_pairs

    @staticmethod
    def _make_date(cell_string: str) -> Date:
        string_parts = cell_string.split("_")
        year = -1
        month = -1
        day = -1
        for part in string_parts:
            if part.isdigit():
                if len(part) == 4:
                    year = int(part)
                else:
                    day = int(part)
            elif part in MONTH_NUMBERS:
                month = MONTH_NUMBERS[part]
        return Date(year, month, day)

    def _values_look_like_dates(self, expression_list: NestedList) -> bool:
        # TODO (pradeep): This will be unnecessary when we have column types identified.
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return False
        # We try to figure out if the values being compared are simple numbers or dates. We use
        # simple rules here: that the string contains less than 4 parts, and one of the parts is a
        # month name. Note that this will not consider strings with just years as dates. That's fine
        # because we can compare them as numbers.
        values_are_dates = False
        # Apply rules on the first cell only, and assume the decision applies to the remaining cells
        # as well.
        cell_value_parts = row_list[0][column_name].replace('fb:cell.', '').split('_')
        # Check if the number of parts in the string are 3 or fewer. If not, it's probably neither a
        # date nor a number.
        if len(cell_value_parts) <= 3:
            for part in cell_value_parts:
                if part in MONTH_NUMBERS:
                    values_are_dates = True
        return values_are_dates

    def _argmax(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows and a column (decoded from the `expression_list`) and returns a list
        containing a single row (dict from columns to cells) that has the maximum numerical value in
        the given column. We return a list instead of a single dict to be consistent with the return
        type of `_select` and `_all_rows`.
        """
        if self._values_look_like_dates(expression_list):
            date_row_pairs = self._get_date_row_pairs_to_filter(expression_list)
            if not date_row_pairs:
                return []
            # Returns a list containing the row with the max cell value.
            return [sorted(date_row_pairs, reverse=True)[0][1]]
        else:
            number_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
            if not number_row_pairs:
                return []
            # Returns a list containing the row with the max cell value.
            return [sorted(number_row_pairs, reverse=True)[0][1]]

    def _argmin(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows and a column (decoded from the `expression_list`) and returns a list
        containing a single row (dict from columns to cells) that has the minimum numerical value in
        the given column. We return a list instead of a single dict to be consistent with the return
        type of `_select` and `_all_rows`.
        """
        if self._values_look_like_dates(expression_list):
            date_row_pairs = self._get_date_row_pairs_to_filter(expression_list)
            if not date_row_pairs:
                return []
            # Returns a list containing the row with the min cell value.
            return [sorted(date_row_pairs)[0][1]]
        else:
            number_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
            if not number_row_pairs:
                return []
            # Returns a list containing the row with the min cell value.
            return [sorted(number_row_pairs)[0][1]]

    def _filter_number_greater(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column is greater than the given value.
        """
        return_list = []
        cell_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, float):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value > filter_value:
                return_list.append(row)
        return return_list

    def _filter_number_greater_equals(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column is greater than or equal to the given value.
        """
        return_list = []
        cell_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, float):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value >= filter_value:
                return_list.append(row)
        return return_list

    def _filter_number_lesser(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column is lesser than the given value.
        """
        return_list = []
        cell_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, float):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value < filter_value:
                return_list.append(row)
        return return_list

    def _filter_number_lesser_equals(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column is lesser than or equal to the given value.
        """
        return_list = []
        cell_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, float):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value <= filter_value:
                return_list.append(row)
        return return_list

    def _filter_number_equals(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column equals the given value.
        """
        return_list = []
        cell_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, float):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value == filter_value:
                return_list.append(row)
        return return_list

    def _filter_number_not_equals(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column is not equal to the given value.
        """
        return_list = []
        cell_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, float):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value != filter_value:
                return_list.append(row)
        return return_list

    # Note that the following six methods are identical to the ones above, except that the filter
    # values are obtained from `_get_date_row_pairs_to_filter`.
    def _filter_date_greater(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a date value (decoded from `expression_list`) and
        returns all the rows where the value in that column is greater than the given value.
        """
        return_list = []
        cell_row_pairs = self._get_date_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, Date):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value > filter_value:
                return_list.append(row)
        return return_list

    def _filter_date_greater_equals(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a date value (decoded from `expression_list`) and
        returns all the rows where the value in that column is greater than or equal to the given value.
        """
        return_list = []
        cell_row_pairs = self._get_date_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, Date):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value >= filter_value:
                return_list.append(row)
        return return_list

    def _filter_date_lesser(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a date value (decoded from `expression_list`) and
        returns all the rows where the value in that column is lesser than the given value.
        """
        return_list = []
        cell_row_pairs = self._get_date_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, Date):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value < filter_value:
                return_list.append(row)
        return return_list

    def _filter_date_lesser_equals(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a date value (decoded from `expression_list`) and
        returns all the rows where the value in that column is lesser than or equal to the given value.
        """
        return_list = []
        cell_row_pairs = self._get_date_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, Date):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value <= filter_value:
                return_list.append(row)
        return return_list

    def _filter_date_equals(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a date value (decoded from `expression_list`) and
        returns all the rows where the value in that column equals the given value.
        """
        return_list = []
        cell_row_pairs = self._get_date_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, Date):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value == filter_value:
                return_list.append(row)
        return return_list

    def _filter_date_not_equals(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a date value (decoded from `expression_list`) and
        returns all the rows where the value in that column is not equal to the given value.
        """
        return_list = []
        cell_row_pairs = self._get_date_row_pairs_to_filter(expression_list)
        filter_value = self._handle_expression(expression_list[2])
        if not isinstance(filter_value, Date):
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        for cell_value, row in cell_row_pairs:
            if cell_value != filter_value:
                return_list.append(row)
        return return_list

    def _filter_in(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a string value (decoded from `expression_list`) and
        returns all the rows where the value in that column contains the given string.
        """
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        # Assuming filter value is a simple string with underscores for spaces. The cell values also
        # have underscores for spaces, so we do not need to replace them here.
        filter_value = str(expression_list[2])
        result_list = []
        for row in row_list:
            if filter_value in row[column_name].replace("fb:cell.", ""):
                result_list.append(row)
        return result_list

    def _filter_not_in(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a string value (decoded from `expression_list`) and
        returns all the rows where the value in that column does not contain the given string.
        """
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        # Assuming filter value is a simple string with underscores for spaces. The cell values also
        # have underscores for spaces, so we do not need to replace them here.
        filter_value = str(expression_list[2])
        result_list = []
        for row in row_list:
            if filter_value not in row[column_name].replace("fb:cell.", ""):
                result_list.append(row)
        return result_list

    def _first(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes an expression that evaluates to a list of rows, and returns the first one in that
        list.
        """
        row_list: List[Dict[str, str]] = self._handle_expression(expression_list)
        if not row_list:
            logger.warning("Trying to get first row from an empty list: %s", expression_list)
            return []
        return [row_list[0]]

    def _last(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes an expression that evaluates to a list of rows, and returns the last one in that
        list.
        """
        row_list: List[Dict[str, str]] = self._handle_expression(expression_list)
        if not row_list:
            logger.warning("Trying to get last row from an empty list: %s", expression_list)
            return []
        return [row_list[-1]]

    def _get_row_index(self, row: Dict[str, str]) -> int:
        """
        Takes a row and returns its index in the full list of rows. If the row does not occur in the
        table (which should never happen because this function will only be called with a row that
        is the result of applying one or more functions on the table rows), the method returns -1.
        """
        row_index = -1
        for index, table_row in enumerate(self._table_data):
            if table_row == row:
                row_index = index
                break
        return row_index

    def _previous(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes an expression that evaluates to a single row, and returns the row (as a list to be
        consistent with the rest of the API), that occurs before the input row in the original set
        of rows. If the input row happens to be the top row, we will return an empty list.
        """
        row_list: List[Dict[str, str]] = self._handle_expression(expression_list)
        if not row_list:
            logger.warning("Trying to get the previous row from an empty list: %s", expression_list)
            return []
        if len(row_list) > 1:
            logger.warning("Trying to get the previous row from a non-singleton list: %s", expression_list)
        input_row_index = self._get_row_index(row_list[0])  # Take the first row.
        if input_row_index > 0:
            return [self._table_data[input_row_index - 1]]
        return []

    def _next(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes an expression that evaluates to a single row, and returns the row (as a list to be
        consistent with the rest of the API), that occurs after the input row in the original set
        of rows. If the input row happens to be the last row, we will return an empty list.
        """
        row_list: List[Dict[str, str]] = self._handle_expression(expression_list)
        if not row_list:
            logger.warning("Trying to get the next row from an empty list: %s", expression_list)
            return []
        if len(row_list) > 1:
            logger.warning("Trying to get the next row from a non-singleton list: %s", expression_list)
        input_row_index = self._get_row_index(row_list[-1])  # Take the last row.
        if input_row_index < len(self._table_data) - 1 and input_row_index != -1:
            return [self._table_data[input_row_index + 1]]
        return []

    def _count(self, expression_list: NestedList) -> float:
        """
        Takes an expression that evaluates to a a list of rows and returns their count (as a float
        to be consistent with the other functions like max that also return numbers).
        """
        row_list: List[Dict[str, str]] = self._handle_expression(expression_list)
        return float(len(row_list))

    def _max(self, expression_list: NestedList) -> float:
        """
        Takes an expression list that evaluates to a  list of rows and a column, and returns the max
        of the values under that column in those rows.
        """
        cell_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
        if not cell_row_pairs:
            return 0.0
        return max([value for value, _ in cell_row_pairs])

    def _min(self, expression_list: NestedList) -> float:
        """
        Takes an expression list that evaluates to a  list of rows and a column, and returns the min
        of the values under that column in those rows.
        """
        cell_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
        if not cell_row_pairs:
            return 0.0
        return min([value for value, _ in cell_row_pairs])

    def _sum(self, expression_list: NestedList) -> float:
        """
        Takes an expression list that evaluates to a  list of rows and a column, and returns the sum
        of the values under that column in those rows.
        """
        cell_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
        if not cell_row_pairs:
            return 0.0
        return sum([value for value, _ in cell_row_pairs])

    def _average(self, expression_list: NestedList) -> float:
        """
        Takes an expression list that evaluates to a  list of rows and a column, and returns the mean
        of the values under that column in those rows.
        """
        cell_row_pairs = self._get_number_row_pairs_to_filter(expression_list)
        if not cell_row_pairs:
            return 0.0
        return sum([value for value, _ in cell_row_pairs]) / len(cell_row_pairs)

    def _mode(self, expression_list: NestedList) -> List[str]:
        """
        Takes an expression that evaluates to a list of rows and a column, and returns the most
        frequent values (one or more) under that column in those rows.
        """
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        value_frequencies: Dict[str, int] = defaultdict(int)
        max_frequency = 0
        most_frequent_list: List[str] = []
        for row in row_list:
            cell_value = row[column_name]
            value_frequencies[cell_value] += 1
            frequency = value_frequencies[cell_value]
            if frequency > max_frequency:
                max_frequency = frequency
                most_frequent_list = [cell_value]
            elif frequency == max_frequency:
                most_frequent_list.append(cell_value)
        return most_frequent_list

    def _same_as(self, expression_list: NestedList) -> List[Dict[str, str]]:
        """
        Takes an expression that evaluates to a row and a column, and returns a list of rows from
        the full set of rows that contain the same value under the given column as the given row.
        """
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        if len(row_list) > 1:
            logger.warning("same_as function got multiple rows. Taking the first one: "
                           f"{expression_list[0]}")
        cell_value = row_list[0][column_name]
        return_list = []
        for row in self._table_data:
            if row[column_name] == cell_value:
                return_list.append(row)
        return return_list

    def _diff(self, expression_list: NestedList) -> float:
        """
        Takes an expression that evaluates to two rows and a column, and returns the difference
        between the values under that column in those two rows.
        """
        first_row_list = self._handle_expression(expression_list[0])
        second_row_list = self._handle_expression(expression_list[1])
        column_name = expression_list[2]
        if not first_row_list or not second_row_list:
            return 0.0
        if len(first_row_list) > 1:
            logger.warning("diff got multiple rows for first argument. Taking the first one: "
                           f"{expression_list[0]}")
        if len(second_row_list) > 1:
            logger.warning("diff got multiple rows for second argument. Taking the first one: "
                           f"{expression_list[1]}")
        first_row = first_row_list[0]
        second_row = second_row_list[0]
        try:
            first_value = float(first_row[column_name].replace("fb:cell.", ""))
            second_value = float(second_row[column_name].replace("fb:cell.", ""))
            return first_value - second_value
        except ValueError:
            logger.error("Invalid column for diff: %s", column_name)
            raise ExecutionError(f"Invalid column for diff: {column_name}")

    @staticmethod
    def _date(expression_list: NestedList) -> Date:
        """
        Takes an expression that evaluates to three numbers, and returns a ``Date`` object whose
        year, month, and day are the three numbers in that order.
        """
        try:
            year = int(str(expression_list[0]))
            month = int(str(expression_list[1]))
            day = int(str(expression_list[2]))
            return Date(year, month, day)
        except ValueError:
            logger.error("Invalid date: %s", expression_list)
            raise ExecutionError(f"Invalid date: {expression_list}")
