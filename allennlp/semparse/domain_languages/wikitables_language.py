from collections import defaultdict
# We use "Number" in a bunch of places throughout to try to generalize ints and floats.
# Unfortunately, mypy doesn't like this very much, so we have to "type: ignore" a bunch of things.
# But it makes for a nicer induced grammar, so it's worth it.
from numbers import Number
from typing import Dict, List, NamedTuple, Set, Tuple
import logging
import re

from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, ExecutionError,
                                                                PredicateType, predicate)
from allennlp.semparse.contexts.table_question_knowledge_graph import MONTH_NUMBERS
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.tools import wikitables_evaluator as evaluator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Date:
    def __init__(self, year: int, month: int, day: int) -> None:
        self.year = year
        self.month = month
        self.day = day

    def __eq__(self, other) -> bool:
        # Note that the logic below renders equality to be non-transitive. That is,
        # Date(2018, -1, -1) == Date(2018, 2, 3) and Date(2018, -1, -1) == Date(2018, 4, 5)
        # but Date(2018, 2, 3) != Date(2018, 4, 5).
        if not isinstance(other, Date):
            raise ExecutionError("Can only compare Dates with Dates")
        year_is_same = self.year == -1 or other.year == -1 or self.year == other.year
        month_is_same = self.month == -1 or other.month == -1 or self.month == other.month
        day_is_same = self.day == -1 or other.day == -1 or self.day == other.day
        return year_is_same and month_is_same and day_is_same

    def __gt__(self, other) -> bool:
        # pylint: disable=too-many-return-statements
        # The logic below is tricky, and is based on some assumptions we make about date comparison.
        # Year, month or day being -1 means that we do not know its value. In those cases, the
        # we consider the comparison to be undefined, and return False if all the fields that are
        # more significant than the field being compared are equal. However, when year is -1 for both
        # dates being compared, it is safe to assume that the year is not specified because it is
        # the same. So we make an exception just in that case. That is, we deem the comparison
        # undefined only when one of the year values is -1, but not both.
        if not isinstance(other, Date):
            raise ExecutionError("Can only compare Dates with Dates")
        # We're doing an exclusive or below.
        if (self.year == -1) != (other.year == -1):
            return False  # comparison undefined
        # If both years are -1, we proceed.
        if self.year != other.year:
            return self.year > other.year
        # The years are equal and not -1, or both are -1.
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
            raise ExecutionError("Can only compare Dates with Dates")
        return self > other or self == other

    def __str__(self):
        return f"{self.year}-{self.month}-{self.day}"


class Row(NamedTuple):
    # Maps column names to cell values.
    values: Dict[str, str]


class Column(NamedTuple):
    name: str


class StringColumn(Column):
    pass


class ComparableColumn(Column):
    pass


class DateColumn(ComparableColumn):
    pass


class NumberColumn(ComparableColumn):
    pass


# TODO(mattg, pradeep): We should probably coerce the question strings to have type `List[str]`, so
# that the filter_in and filter_not_in functions can easily use the output of a `select` as input.
# Also, there are updates to this language in `allennlp-weak-supervision-research` that need to be
# included here.

class WikiTablesLanguage(DomainLanguage):
    # pylint: disable=too-many-public-methods,no-self-use
    """
    Implements the functions in the variable free language we use, that's inspired by the one in
    "Memory Augmented Policy Optimization for Program Synthesis with Generalization" by Liang et al.

    Because some of the functions are only allowed if some conditions hold on the table, we don't
    use the ``@predicate`` decorator for all of the language functions.  Instead, we add them to
    the language using ``add_predicate`` if, e.g., there is a column with dates in it.
    """
    def __init__(self, table_context: TableQuestionContext) -> None:
        super().__init__(start_types={Number, Date, List[str]})
        self.table_context = table_context
        self.table_data = [Row(row) for row in table_context.table_data]

        column_types = table_context.column_types.values()
        if "string" in column_types:
            self.add_predicate('filter_in', self.filter_in)
            self.add_predicate('filter_not_in', self.filter_not_in)
        if "date" in column_types:
            self.add_predicate('filter_date_greater', self.filter_date_greater)
            self.add_predicate('filter_date_greater_equals', self.filter_date_greater_equals)
            self.add_predicate('filter_date_lesser', self.filter_date_lesser)
            self.add_predicate('filter_date_lesser_equals', self.filter_date_lesser_equals)
            self.add_predicate('filter_date_equals', self.filter_date_equals)
            self.add_predicate('filter_date_not_equals', self.filter_date_not_equals)
            # Adding -1 to mapping because we need it for dates where not all three fields are
            # specified. We want to do this only when the table has a date column. This is because
            # the knowledge graph is also constructed in such a way that -1 is an entity with date
            # columns as the neighbors only if any date columns exist in the table.
            self.add_constant('-1', -1, type_=Number)
        if "number" in column_types:
            self.add_predicate('filter_number_greater', self.filter_number_greater)
            self.add_predicate('filter_number_greater_equals', self.filter_number_greater_equals)
            self.add_predicate('filter_number_lesser', self.filter_number_lesser)
            self.add_predicate('filter_number_lesser_equals', self.filter_number_lesser_equals)
            self.add_predicate('filter_number_equals', self.filter_number_equals)
            self.add_predicate('filter_number_not_equals', self.filter_number_not_equals)
            self.add_predicate('max', self.max)
            self.add_predicate('min', self.min)
            self.add_predicate('average', self.average)
            self.add_predicate('sum', self.sum)
            self.add_predicate('diff', self.diff)
        if "date" in column_types or "number" in column_types:
            self.add_predicate('argmax', self.argmax)
            self.add_predicate('argmin', self.argmin)

        self.table_graph = table_context.get_table_knowledge_graph()

        # Adding entities and numbers seen in questions as constants.
        question_entities, question_numbers = table_context.get_entities_from_question()
        self._question_entities = [entity for entity, _ in question_entities]
        self._question_numbers = [number for number, _ in question_numbers]
        for entity in self._question_entities:
            self.add_constant(entity, entity)

        for number in self._question_numbers:
            self.add_constant(str(number), float(number), type_=Number)

        # Keeps track of column name productions so that we can add them to the agenda.
        self._column_productions_for_agenda: Dict[str, str] = {}

        # Adding column names as constants.  Each column gets added once for every
        # type in the hierarchy going from its concrete class to the base Column.  String columns
        # get added as StringColumn and Column, and date and number columns get added as DateColumn
        # (or NumberColumn), ComparableColumn, and Column.
        for column_name, column_type in table_context.column_types.items():
            column_name = f"{column_type}_column:{column_name}"
            column: Column = None
            if column_type == 'string':
                column = StringColumn(column_name)
            elif column_type == 'date':
                column = DateColumn(column_name)
                self.add_constant(column_name, column, type_=ComparableColumn)
            elif column_type == 'number':
                column = NumberColumn(column_name)
                self.add_constant(column_name, column, type_=ComparableColumn)
            self.add_constant(column_name, column, type_=Column)
            self.add_constant(column_name, column)
            column_type_name = str(PredicateType.get_type(type(column)))
            self._column_productions_for_agenda[column_name] = f"{column_type_name} -> {column_name}"

        # Mapping from terminal strings to productions that produce them.  We use this in the
        # agenda-related methods, and some models that use this language look at this field to know
        # how many terminals to plan for.
        self.terminal_productions: Dict[str, str] = {}
        for name, types in self._function_types.items():
            self.terminal_productions[name] = "%s -> %s" % (types[0], name)

    def get_agenda(self):
        agenda_items = []
        question_tokens = [token.text for token in self.table_context.question_tokens]
        question = " ".join(question_tokens)
        if "at least" in question:
            agenda_items.append("filter_number_greater_equals")
        if "at most" in question:
            agenda_items.append("filter_number_lesser_equals")

        comparison_triggers = ["greater", "larger", "more"]
        if any("no %s than" %word in question for word in comparison_triggers):
            agenda_items.append("filter_number_lesser_equals")
        elif any("%s than" %word in question for word in comparison_triggers):
            agenda_items.append("filter_number_greater")
        for token in question_tokens:
            if token in ["next", "after", "below"]:
                agenda_items.append("next")
            if token in ["previous", "before", "above"]:
                agenda_items.append("previous")
            if token == "total":
                agenda_items.append("sum")
            if token == "difference":
                agenda_items.append("diff")
            if token == "average":
                agenda_items.append("average")
            if token in ["least", "smallest", "shortest", "lowest"] and "at least" not in question:
                # This condition is too brittle. But for most logical forms with "min", there are
                # semantically equivalent ones with "argmin". The exceptions are rare.
                if "what is the least" in question:
                    agenda_items.append("min")
                else:
                    agenda_items.append("argmin")
            if token in ["most", "largest", "highest", "longest", "greatest"] and "at most" not in question:
                # This condition is too brittle. But for most logical forms with "max", there are
                # semantically equivalent ones with "argmax". The exceptions are rare.
                if "what is the most" in question:
                    agenda_items.append("max")
                else:
                    agenda_items.append("argmax")
            if token in ["first", "top"]:
                agenda_items.append("first")
            if token in ["last", "bottom"]:
                agenda_items.append("last")

        if "how many" in question:
            if "sum" not in agenda_items and "average" not in agenda_items:
                # The question probably just requires counting the rows. But this is not very
                # accurate. The question could also be asking for a value that is in the table.
                agenda_items.append("count")
        agenda = []
        # Adding productions from the global set.
        for agenda_item in set(agenda_items):
            # Some agenda items may not be present in the terminal productions because some of these
            # terminals are table-content specific. For example, if the question triggered "sum",
            # and the table does not have number columns, we should not add "<r,<f,n>> -> sum" to
            # the agenda.
            if agenda_item in self.terminal_productions:
                agenda.append(self.terminal_productions[agenda_item])

        # Adding column names that occur in question.
        question_with_underscores = "_".join(question_tokens)
        normalized_question = re.sub("[^a-z0-9_]", "", question_with_underscores)
        # We keep track of tokens that are in column names being added to the agenda. We will not
        # add string productions to the agenda if those tokens were already captured as column
        # names.
        # Note: If the same string occurs multiple times, this may cause string productions being
        # omitted from the agenda unnecessarily. That is fine, as we want to err on the side of
        # adding fewer rules to the agenda.
        tokens_in_column_names: Set[str] = set()
        for column_name_with_type, signature in self._column_productions_for_agenda.items():
            column_name = column_name_with_type.split(":")[1]
            # Underscores ensure that the match is of whole words.
            if f"_{column_name}_" in normalized_question:
                agenda.append(signature)
                for token in column_name.split("_"):
                    tokens_in_column_names.add(token)

        # Adding all productions that lead to entities and numbers extracted from the question.
        for entity in self._question_entities:
            if entity.replace("string:", "") not in tokens_in_column_names:
                agenda.append(f"str -> {entity}")

        for number in self._question_numbers:
            # The reason we check for the presence of the number in the question again is because
            # some of these numbers are extracted from number words like month names and ordinals
            # like "first". On looking at some agenda outputs, I found that they hurt more than help
            # in the agenda.
            if f"_{number}_" in normalized_question:
                agenda.append(f"Number -> {number}")
        return agenda

    def evaluate_logical_form(self, logical_form: str, target_list: List[str]) -> bool:
        """
        Takes a logical form, and the list of target values as strings from the original lisp
        string, and returns True iff the logical form executes to the target list, using the
        official WikiTableQuestions evaluation script.
        """
        normalized_target_list = [TableQuestionContext.normalize_string(value) for value in
                                  target_list]
        target_value_list = evaluator.to_value_list(normalized_target_list)
        try:
            denotation = self.execute(logical_form)
        except ExecutionError:
            logger.warning(f'Failed to execute: {logical_form}')
            return False
        if isinstance(denotation, list):
            denotation_list = [str(denotation_item) for denotation_item in denotation]
        else:
            denotation_list = [str(denotation)]
        denotation_value_list = evaluator.to_value_list(denotation_list)
        return evaluator.check_denotation(target_value_list, denotation_value_list)

    # Things below here are language predicates, until you get to private methods.  We start with
    # general predicates that are always included in the language, then move to
    # column-type-specific predicates, which only get added if we see columns of particular types
    # in the table.

    @predicate
    def all_rows(self) -> List[Row]:
        return self.table_data

    @predicate
    def select(self, rows: List[Row], column: Column) -> List[str]:
        """
        Select function takes a list of rows and a column and returns a list of cell values as
        strings.
        """
        return [row.values[column.name] for row in rows]

    @predicate
    def same_as(self, rows: List[Row], column: Column) -> List[Row]:
        """
        Takes a row and a column and returns a list of rows from the full set of rows that contain
        the same value under the given column as the given row.
        """
        cell_value = rows[0].values[column.name]
        return_list = []
        for table_row in self.table_data:
            if table_row.values[column.name] == cell_value:
                return_list.append(table_row)
        return return_list

    @predicate
    def date(self, year: Number, month: Number, day: Number) -> Date:
        """
        Takes three numbers and returns a ``Date`` object whose year, month, and day are the three
        numbers in that order.
        """
        return Date(year, month, day)  # type: ignore

    @predicate
    def first(self, rows: List[Row]) -> List[Row]:
        """
        Takes an expression that evaluates to a list of rows, and returns the first one in that
        list.
        """
        if not rows:
            logger.warning("Trying to get first row from an empty list")
            return []
        return [rows[0]]

    @predicate
    def last(self, rows: List[Row]) -> List[Row]:
        """
        Takes an expression that evaluates to a list of rows, and returns the last one in that
        list.
        """
        if not rows:
            logger.warning("Trying to get first row from an empty list")
            return []
        return [rows[-1]]

    @predicate
    def previous(self, rows: List[Row]) -> List[Row]:
        """
        Takes an expression that evaluates to a single row, and returns the row that occurs before
        the input row in the original set of rows. If the input row happens to be the top row, we
        will return an empty list.
        """
        if not rows:
            return []
        input_row_index = self._get_row_index(rows[0])
        if input_row_index > 0:
            return [self.table_data[input_row_index - 1]]
        return []

    @predicate
    def next(self, rows: List[Row]) -> List[Row]:
        """
        Takes an expression that evaluates to a single row, and returns the row that occurs after
        the input row in the original set of rows. If the input row happens to be the last row, we
        will return an empty list.
        """
        if not rows:
            return []
        input_row_index = self._get_row_index(rows[0])
        if input_row_index < len(self.table_data) - 1 and input_row_index != -1:
            return [self.table_data[input_row_index + 1]]
        return []

    @predicate
    def count(self, rows: List[Row]) -> Number:
        return len(rows)  # type: ignore

    @predicate
    def mode(self, rows: List[Row], column: Column) -> List[str]:
        """
        Takes a list of rows and a column and returns the most frequent values (one or more) under
        that column in those rows.
        """
        value_frequencies: Dict[str, int] = defaultdict(int)
        max_frequency = 0
        most_frequent_list: List[str] = []
        for row in rows:
            cell_value = row.values[column.name]
            value_frequencies[cell_value] += 1
            frequency = value_frequencies[cell_value]
            if frequency > max_frequency:
                max_frequency = frequency
                most_frequent_list = [cell_value]
            elif frequency == max_frequency:
                most_frequent_list.append(cell_value)
        return most_frequent_list

    # These get added to the language (using `add_predicate()`) if we see a date or number column
    # in the table.

    def argmax(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        # We just check whether the first cell value is a date or number and assume that the rest
        # are the same kind of values.
        first_cell_value = rows[0].values[column.name]
        if self._value_looks_like_date(first_cell_value):
            value_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        else:
            value_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)  # type: ignore
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]]

    def argmin(self, rows: List[Row], column: ComparableColumn) -> List[Row]:
        """
        Takes a list of rows and a column and returns a list containing a single row (dict from
        columns to cells) that has the minimum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            return []
        # We just check whether the first cell value is a date or number and assume that the rest
        # are the same kind of values.
        first_cell_value = rows[0].values[column.name]
        if self._value_looks_like_date(first_cell_value):
            value_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        else:
            value_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)  # type: ignore
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0])[0][1]]

    # These six methods take a list of rows, a column, and a numerical value and return all the
    # rows where the value in that column is [comparator] than the given value.  They only get
    # added to the language if we see a number column in the table.

    def filter_number_greater(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value > filter_value]  # type: ignore

    def filter_number_greater_equals(self,
                                     rows: List[Row],
                                     column: NumberColumn,
                                     filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value >= filter_value]  # type: ignore

    def filter_number_lesser(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value < filter_value]  # type: ignore

    def filter_number_lesser_equals(self,
                                    rows: List[Row],
                                    column: NumberColumn,
                                    filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value <= filter_value]  # type: ignore

    def filter_number_equals(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value == filter_value]  # type: ignore

    def filter_number_not_equals(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value != filter_value]  # type: ignore

    # These six methods are the same as the six above, but for dates.  They only get added to the
    # language if we see a date column in the table.

    def filter_date_greater(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value > filter_value]

    def filter_date_greater_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value >= filter_value]

    def filter_date_lesser(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value < filter_value]

    def filter_date_lesser_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value <= filter_value]

    def filter_date_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value == filter_value]

    def filter_date_not_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value != filter_value]

    # These two are similar to the filter methods above, but operate on strings obtained from the
    # question, instead of dates or numbers.  So they check for whether the string value is present
    # in the cell or not, instead of using a numerical / date comparator.  We only add them to the
    # language if we see a string column in the table (which is basically always).

    def filter_in(self, rows: List[Row], column: StringColumn, filter_value: str) -> List[Row]:
        # Assuming filter value has underscores for spaces. The cell values also have underscores
        # for spaces, so we do not need to replace them here.  Also, we need to remove the
        # "string:" that was prepended to the entity name in the language.
        filter_value = filter_value.lstrip('string:')
        return [row for row in rows if filter_value in row.values[column.name]]

    def filter_not_in(self, rows: List[Row], column: StringColumn, filter_value: str) -> List[Row]:
        # Assuming filter value has underscores for spaces. The cell values also have underscores
        # for spaces, so we do not need to replace them here.  Also, we need to remove the
        # "string:" that was prepended to the entity name in the language.
        filter_value = filter_value.lstrip('string:')
        return [row for row in rows if filter_value not in row.values[column.name]]

    # These are some more number-column-specific functions, which only get added if we see a number
    # column.

    def max(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the max of the values under that column in
        those rows.
        """
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        if not cell_row_pairs:
            return 0.0  # type: ignore
        return max([value for value, _ in cell_row_pairs])  # type: ignore

    def min(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the min of the values under that column in
        those rows.
        """
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        if not cell_row_pairs:
            return 0.0  # type: ignore
        return min([value for value, _ in cell_row_pairs])  # type: ignore

    def sum(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the sum of the values under that column in
        those rows.
        """
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        if not cell_row_pairs:
            return 0.0  # type: ignore
        return sum([value for value, _ in cell_row_pairs])  # type: ignore

    def average(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the mean of the values under that column in
        those rows.
        """
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        if not cell_row_pairs:
            return 0.0  # type: ignore
        return sum([value for value, _ in cell_row_pairs]) / len(cell_row_pairs)  # type: ignore

    def diff(self, first_row: List[Row], second_row: List[Row], column: NumberColumn) -> Number:
        """
        Takes a two rows and a number column and returns the difference between the values under
        that column in those two rows.
        """
        if not first_row or not second_row:
            return 0.0  # type: ignore
        try:
            first_value = float(first_row[0].values[column.name])
            second_value = float(second_row[0].values[column.name])
            return first_value - second_value  # type: ignore
        except ValueError:
            raise ExecutionError(f"Invalid column for diff: {column.name}")

    # End of language predicates.  Stuff below here is for private use, helping to implement the
    # functions above.

    def __eq__(self, other):
        if not isinstance(other, WikiTablesLanguage):
            return False
        return self.table_data == other.table_data

    @staticmethod
    def _get_number_row_pairs_to_filter(rows: List[Row],
                                        column_name: str) -> List[Tuple[float, Row]]:
        """
        Helper method that takes a row list and a column name, and returns a list of tuples, each
        containing as the first element a number taken from that column, and the corresponding row
        as the second element. The output can be used to compare rows based on the numbers.
        """
        if not rows:
            return []
        try:
            # Various symbols like commas, dollar signs would have been converted to _. Removing
            # them for float conversion.
            cell_row_pairs = [(float(row.values[column_name].replace('_', '')), row) for row in rows]
        except ValueError:
            # This means that at least one of the cells is not numerical.
            return []
        return cell_row_pairs

    def _get_date_row_pairs_to_filter(self,
                                      rows: List[Row],
                                      column_name: str) -> List[Tuple[Date, Row]]:
        """
        Helper method that takes a row list and a column name, and returns a list of tuples, each
        containing as the first element a date taken from that column, and the corresponding row as
        the second element. The output can be used to compare rows based on the dates.
        """
        if not rows:
            return []
        cell_row_pairs = [(self._make_date(row.values[column_name]), row) for row in rows]
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

    @staticmethod
    def _value_looks_like_date(cell_value: str) -> bool:
        # We try to figure out if the values being compared are simple numbers or dates. We use
        # simple rules here: that the string contains less than 4 parts, and one of the parts is a
        # month name. Note that this will not consider strings with just years as dates. That's fine
        # because we can compare them as numbers.
        values_are_dates = False
        cell_value_parts = cell_value.split('_')
        # Check if the number of parts in the string are 3 or fewer. If not, it's probably neither a
        # date nor a number.
        if len(cell_value_parts) <= 3:
            for part in cell_value_parts:
                if part in MONTH_NUMBERS:
                    values_are_dates = True
        return values_are_dates

    def _get_row_index(self, row: Row) -> int:
        """
        Takes a row and returns its index in the full list of rows. If the row does not occur in the
        table (which should never happen because this function will only be called with a row that
        is the result of applying one or more functions on the table rows), the method returns -1.
        """
        row_index = -1
        for index, table_row in enumerate(self.table_data):
            if table_row.values == row.values:
                row_index = index
                break
        return row_index
