from collections import defaultdict
# We use "Number" in a bunch of places throughout to try to generalize ints and floats.
# Unfortunately, mypy doesn't like this very much, so we have to "type: ignore" a bunch of things.
# But it makes for a nicer induced grammar, so it's worth it.
from numbers import Number
from typing import Dict, List, NamedTuple, Set, Type, Tuple, Any
import logging
import re

from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, ExecutionError,
                                                                PredicateType, predicate)
from allennlp.semparse.contexts.table_question_context import MONTH_NUMBERS
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.contexts.table_question_context import CellValueType
from allennlp.semparse.common import Date
from allennlp.tools import wikitables_evaluator as evaluator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Row(NamedTuple):
    # Maps column names to cell values.
    values: Dict[str, CellValueType]


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
        super().__init__(start_types=self._get_start_types_in_context(table_context))
        self.table_context = table_context
        self.table_data = [Row(row) for row in table_context.table_data]

        column_types = table_context.column_types
        self._table_has_string_columns = False
        self._table_has_date_columns = False
        self._table_has_number_columns = False
        if "string" in column_types:
            self.add_predicate('filter_in', self.filter_in)
            self.add_predicate('filter_not_in', self.filter_not_in)
            self._table_has_string_columns = True
        if "date" in column_types:
            self.add_predicate('filter_date_greater', self.filter_date_greater)
            self.add_predicate('filter_date_greater_equals', self.filter_date_greater_equals)
            self.add_predicate('filter_date_lesser', self.filter_date_lesser)
            self.add_predicate('filter_date_lesser_equals', self.filter_date_lesser_equals)
            self.add_predicate('filter_date_equals', self.filter_date_equals)
            self.add_predicate('filter_date_not_equals', self.filter_date_not_equals)
            self.add_predicate('max_date', self.max_date)
            self.add_predicate('min_date', self.min_date)
            # Adding -1 to mapping because we need it for dates where not all three fields are
            # specified. We want to do this only when the table has a date column. This is because
            # the knowledge graph is also constructed in such a way that -1 is an entity with date
            # columns as the neighbors only if any date columns exist in the table.
            self.add_constant('-1', -1, type_=Number)
            self._table_has_date_columns = True
        if "number" in column_types or "num2" in column_types:
            self.add_predicate('filter_number_greater', self.filter_number_greater)
            self.add_predicate('filter_number_greater_equals', self.filter_number_greater_equals)
            self.add_predicate('filter_number_lesser', self.filter_number_lesser)
            self.add_predicate('filter_number_lesser_equals', self.filter_number_lesser_equals)
            self.add_predicate('filter_number_equals', self.filter_number_equals)
            self.add_predicate('filter_number_not_equals', self.filter_number_not_equals)
            self.add_predicate('max_number', self.max_number)
            self.add_predicate('min_number', self.min_number)
            self.add_predicate('average', self.average)
            self.add_predicate('sum', self.sum)
            self.add_predicate('diff', self.diff)
            self._table_has_number_columns = True
        if "date" in column_types or "number" in column_types or "num2" in column_types:
            self.add_predicate('argmax', self.argmax)
            self.add_predicate('argmin', self.argmin)

        self.table_graph = table_context.get_table_knowledge_graph()

        # Adding entities and numbers seen in questions as constants.
        question_entities, question_numbers = table_context.get_entities_from_question()
        self._question_entities = [entity for entity, _ in question_entities]
        self._question_numbers = [number for number, _ in question_numbers]
        for entity in self._question_entities:
            # Forcing the type of entities to be List[str] here to ensure that the language deals with the outputs
            # of select-like statements and constants similarly.
            self.add_constant(entity, entity, type_=List[str])

        for number in self._question_numbers:
            self.add_constant(str(number), float(number), type_=Number)

        # Keeps track of column name productions so that we can add them to the agenda.
        self._column_productions_for_agenda: Dict[str, str] = {}

        # Adding column names as constants.
        for column_name in table_context.column_names:
            column_type = column_name.split(":")[0].replace("_column", "")
            column: Column = None
            if column_type == 'string':
                column = StringColumn(column_name)
            elif column_type == 'date':
                column = DateColumn(column_name)
                self.add_constant(column_name, column, type_=ComparableColumn)
            elif column_type == 'number' or column_type == "num2":
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

    def _get_start_types_in_context(self, table_context: TableQuestionContext) -> Set[Type]:
        start_types: Set[Type] = set()
        if "string" in table_context.column_types:
            start_types.add(List[str])
        if "number" in table_context.column_types or "num2" in table_context.column_types:
            start_types.add(Number)
        if "date" in table_context.column_types:
            start_types.add(Date)
        return start_types

    def get_agenda(self,
                   conservative: bool = False):
        """
        Returns an agenda that can be used guide search.

        Parameters
        ----------
        conservative : ``bool``
            Setting this flag will return a subset of the agenda items that correspond to high
            confidence lexical matches. You'll need this if you are going to use this agenda to
            penalize a model for producing logical forms that do not contain some items in it. In
            that case, you'll want this agenda to have close to perfect precision, at the cost of a
            lower recall. You may not want to set this flag if you are sorting the output from a
            search procedure based on how much of this agenda is satisfied.
        """
        agenda_items = []
        question_tokens = [token.text for token in self.table_context.question_tokens]
        question = " ".join(question_tokens)

        added_number_filters = False
        if self._table_has_number_columns:
            if "at least" in question:
                agenda_items.append("filter_number_greater_equals")
            if "at most" in question:
                agenda_items.append("filter_number_lesser_equals")

            comparison_triggers = ["greater", "larger", "more"]
            if any(f"no {word} than" in question for word in comparison_triggers):
                agenda_items.append("filter_number_lesser_equals")
            elif any(f"{word} than" in question for word in comparison_triggers):
                agenda_items.append("filter_number_greater")

            # We want to keep track of this because we do not want to add both number and date
            # filters to the agenda if we want to be conservative.
            if agenda_items:
                added_number_filters = True
        for token in question_tokens:
            if token in ["next", "below"] or (token == "after" and not conservative):
                agenda_items.append("next")
            if token in ["previous", "above"] or (token == "before" and not conservative):
                agenda_items.append("previous")
            if token in ["first", "top"]:
                agenda_items.append("first")
            if token in ["last", "bottom"]:
                agenda_items.append("last")
            if token == "same":
                agenda_items.append("same_as")

            if self._table_has_number_columns:
                # "total" does not always map to an actual summing operation.
                if token == "total" and not conservative:
                    agenda_items.append("sum")
                if token == "difference" or "how many more" in question or "how much more" in question:
                    agenda_items.append("diff")
                if token == "average":
                    agenda_items.append("average")
                if token in ["least", "smallest", "shortest", "lowest"] and "at least" not in question:
                    # This condition is too brittle. But for most logical forms with "min", there are
                    # semantically equivalent ones with "argmin". The exceptions are rare.
                    if "what is the least" not in question:
                        agenda_items.append("argmin")
                if token in ["most", "largest", "highest", "longest", "greatest"] and "at most" not in question:
                    # This condition is too brittle. But for most logical forms with "max", there are
                    # semantically equivalent ones with "argmax". The exceptions are rare.
                    if "what is the most" not in question:
                        agenda_items.append("argmax")

            if self._table_has_date_columns:
                if token in MONTH_NUMBERS or (token.isdigit() and len(token) == 4 and
                                              int(token) < 2100 and int(token) > 1100):
                    # Token is either a month or an year. We'll add date functions.
                    if not added_number_filters or not conservative:
                        if "after" in question_tokens:
                            agenda_items.append("filter_date_greater")
                        elif "before" in question_tokens:
                            agenda_items.append("filter_date_lesser")
                        elif "not" in question_tokens:
                            agenda_items.append("filter_date_not_equals")
                        else:
                            agenda_items.append("filter_date_equals")

            if "what is the least" in question and self._table_has_number_columns:
                agenda_items.append("min_number")
            if "what is the most" in question and self._table_has_number_columns:
                agenda_items.append("max_number")
            if "when" in question_tokens and self._table_has_date_columns:
                if "last" in question_tokens:
                    agenda_items.append("max_date")
                elif "first" in question_tokens:
                    agenda_items.append("min_date")
                else:
                    agenda_items.append("select_date")


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

        if conservative:
            # Some of the columns in the table have multiple types, and thus occur in the KG as
            # different columns. We do not want to add them all to the agenda if their names,
            # because it is unlikely that logical forms use them all. In fact, to be conservative,
            # we won't add any of them. So we'll first identify such column names.
            refined_column_productions: Dict[str, str] = {}
            for column_name, signature in self._column_productions_for_agenda.items():
                column_type, name = column_name.split(":")
                if column_type == "string_column":
                    if f"number_column:{name}" not in self._column_productions_for_agenda and \
                       f"date_column:{name}" not in self._column_productions_for_agenda:
                        refined_column_productions[column_name] = signature

                elif column_type == "number_column":
                    if f"string_column:{name}" not in self._column_productions_for_agenda and \
                       f"date_column:{name}" not in self._column_productions_for_agenda:
                        refined_column_productions[column_name] = signature

                else:
                    if f"string_column:{name}" not in self._column_productions_for_agenda and \
                       f"number_column:{name}" not in self._column_productions_for_agenda:
                        refined_column_productions[column_name] = signature
            # Similarly, we do not want the same spans in the question to be added to the agenda as
            # both string and number productions.
            refined_entities: List[str] = []
            refined_numbers: List[str] = []
            for entity in self._question_entities:
                if entity.replace("string:", "") not in self._question_numbers:
                    refined_entities.append(entity)
            for number in self._question_numbers:
                if f"string:{number}" not in self._question_entities:
                    refined_numbers.append(number)
        else:
            refined_column_productions = dict(self._column_productions_for_agenda)
            refined_entities = list(self._question_entities)
            refined_numbers = list(self._question_numbers)

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
        for column_name_with_type, signature in refined_column_productions.items():
            column_name = column_name_with_type.split(":")[1]
            # Underscores ensure that the match is of whole words.
            if f"_{column_name}_" in normalized_question:
                agenda.append(signature)
                for token in column_name.split("_"):
                    tokens_in_column_names.add(token)

        # Adding all productions that lead to entities and numbers extracted from the question.
        for entity in refined_entities:
            if entity.replace("string:", "") not in tokens_in_column_names:
                agenda.append(f"List[str] -> {entity}")

        for number in refined_numbers:
            # The reason we check for the presence of the number in the question again is because
            # some of these numbers are extracted from number words like month names and ordinals
            # like "first". On looking at some agenda outputs, I found that they hurt more than help
            # in the agenda.
            if f"_{number}_" in normalized_question:
                agenda.append(f"Number -> {number}")
        return agenda

    @staticmethod
    def is_instance_specific_entity(entity_name: str) -> bool:
        """
        Instance specific entities are column names, strings and numbers. Returns True if the entity
        is one of those.
        """
        entity_is_number = False
        try:
            float(entity_name)
            entity_is_number = True
        except ValueError:
            pass
        # Column names start with "*_column:", strings start with "string:"
        return "_column:" in entity_name or entity_name.startswith("string:") or entity_is_number

    def evaluate_logical_form(self, logical_form: str, target_list: List[str]) -> bool:
        """
        Takes a logical form, and the list of target values as strings from the original lisp
        string, and returns True iff the logical form executes to the target list, using the
        official WikiTableQuestions evaluation script.
        """
        try:
            denotation = self.execute(logical_form)
        except ExecutionError as error:
            logger.warning(f'Failed to execute: {logical_form}. Error: {error}')
            return False
        return self.evaluate_denotation(denotation, target_list)

    def evaluate_action_sequence(self, action_sequence: List[str], target_list: List[str]) -> bool:
        """
        Similar to ``evaluate_logical_form`` except that it takes an action sequence instead. The reason this is
        separate is because there is a separate method ``DomainLanguage.execute_action_sequence`` that executes the
        action sequence directly.
        """
        try:
            denotation = self.execute_action_sequence(action_sequence)
        except ExecutionError:
            logger.warning(f'Failed to execute action sequence: {action_sequence}')
            return False
        return self.evaluate_denotation(denotation, target_list)

    def evaluate_denotation(self, denotation: Any, target_list: List[str]) -> bool:
        """
        Compares denotation with a target list and returns whether they are both the same according to the official
        evaluator.
        """
        normalized_target_list = [TableQuestionContext.normalize_string(value) for value in
                                  target_list]
        target_value_list = evaluator.to_value_list(normalized_target_list)
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
    def select_string(self, rows: List[Row], column: StringColumn) -> List[str]:
        """
        Select function takes a list of rows and a column name and returns a list of strings as
        in cells.
        """
        return [str(row.values[column.name]) for row in rows if row.values[column.name] is not None]

    @predicate
    def select_number(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Select function takes a row (as a list) and a column name and returns the number in that
        column. If multiple rows are given, will return the first number that is not None.
        """
        numbers: List[float] = []
        for row in rows:
            cell_value = row.values[column.name]
            if isinstance(cell_value, float):
                numbers.append(cell_value)

        return numbers[0] if numbers else -1  # type: ignore

    @predicate
    def select_date(self, rows: List[Row], column: DateColumn) -> Date:
        """
        Select function takes a row as a list and a column name and returns the date in that column.
        """
        dates: List[Date] = []
        for row in rows:
            cell_value = row.values[column.name]
            if isinstance(cell_value, Date):
                dates.append(cell_value)

        return dates[0] if dates else Date(-1, -1, -1)  # type: ignore

    @predicate
    def same_as(self, rows: List[Row], column: Column) -> List[Row]:
        """
        Takes a row and a column and returns a list of rows from the full set of rows that contain
        the same value under the given column as the given row.
        """
        return_list: List[Row] = []
        if not rows:
            return return_list
        cell_value = rows[0].values[column.name]
        for table_row in self.table_data:
            new_cell_value = table_row.values[column.name]
            if new_cell_value is None or not isinstance(new_cell_value, type(cell_value)):
                continue
            if new_cell_value == cell_value:
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
            logger.warning("Trying to get last row from an empty list")
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
    def mode_string(self, rows: List[Row], column: StringColumn) -> List[str]:
        """
        Takes a list of rows and a column and returns the most frequent values (one or more) under
        that column in those rows.
        """
        most_frequent_list = self._get_most_frequent_values(rows, column)
        if not most_frequent_list:
            return []
        if not all([isinstance(value, str) for value in most_frequent_list]):
            raise ExecutionError(f"Invalid values for mode_string: {most_frequent_list}")
        return most_frequent_list

    @predicate
    def mode_number(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the most frequent value under
        that column in those rows.
        """
        most_frequent_list = self._get_most_frequent_values(rows, column)
        if not most_frequent_list:
            return 0.0  # type: ignore
        most_frequent_value = most_frequent_list[0]
        if not isinstance(most_frequent_value, Number):
            raise ExecutionError(f"Invalid values for mode_number: {most_frequent_value}")
        return most_frequent_value

    @predicate
    def mode_date(self, rows: List[Row], column: DateColumn) -> Date:
        """
        Takes a list of rows and a column and returns the most frequent value under
        that column in those rows.
        """
        most_frequent_list = self._get_most_frequent_values(rows, column)
        if not most_frequent_list:
            return Date(-1, -1, -1)
        most_frequent_value = most_frequent_list[0]
        if not isinstance(most_frequent_value, Date):
            raise ExecutionError(f"Invalid values for mode_date: {most_frequent_value}")
        return most_frequent_value

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
        value_row_pairs = [(row.values[column.name], row) for row in rows if row.values[column.name] is not None]
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
        value_row_pairs = [(row.values[column.name], row) for row in rows if row.values[column.name] is not None]
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(value_row_pairs, key=lambda x: x[0])[0][1]]

    # These six methods take a list of rows, a column, and a numerical value and return all the
    # rows where the value in that column is [comparator] than the given value.  They only get
    # added to the language if we see a number column in the table.

    def filter_number_greater(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = [(row.values[column.name], row) for row in rows if row.values[column.name] is not None]
        return [row for cell_value, row in cell_row_pairs if cell_value > filter_value]  # type: ignore

    def filter_number_greater_equals(self,
                                     rows: List[Row],
                                     column: NumberColumn,
                                     filter_value: Number) -> List[Row]:
        cell_row_pairs = [(row.values[column.name], row) for row in rows if row.values[column.name] is not None]
        return [row for cell_value, row in cell_row_pairs if cell_value >= filter_value]  # type: ignore

    def filter_number_lesser(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = [(row.values[column.name], row) for row in rows if row.values[column.name] is not None]
        return [row for cell_value, row in cell_row_pairs if cell_value < filter_value]  # type: ignore

    def filter_number_lesser_equals(self,
                                    rows: List[Row],
                                    column: NumberColumn,
                                    filter_value: Number) -> List[Row]:
        cell_row_pairs = [(row.values[column.name], row) for row in rows if row.values[column.name] is not None]
        return [row for cell_value, row in cell_row_pairs if cell_value <= filter_value]  # type: ignore

    def filter_number_equals(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = [(row.values[column.name], row) for row in rows if row.values[column.name] is not None]
        return [row for cell_value, row in cell_row_pairs if cell_value == filter_value]  # type: ignore

    def filter_number_not_equals(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = [(row.values[column.name], row) for row in rows if row.values[column.name] is not None]
        return [row for cell_value, row in cell_row_pairs if cell_value != filter_value]  # type: ignore

    # These six methods are the same as the six above, but for dates.  They only get added to the
    # language if we see a date column in the table.

    def filter_date_greater(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs: List[Tuple[Date, Row]] = []
        for row in rows:
            cell_value = row.values[column.name]
            if isinstance(cell_value, Date):
                cell_row_pairs.append((cell_value, row))

        return [row for cell_value, row in cell_row_pairs if cell_value > filter_value]

    def filter_date_greater_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs: List[Tuple[Date, Row]] = []
        for row in rows:
            cell_value = row.values[column.name]
            if isinstance(cell_value, Date):
                cell_row_pairs.append((cell_value, row))
        return [row for cell_value, row in cell_row_pairs if cell_value >= filter_value]

    def filter_date_lesser(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs: List[Tuple[Date, Row]] = []
        for row in rows:
            cell_value = row.values[column.name]
            if isinstance(cell_value, Date):
                cell_row_pairs.append((cell_value, row))
        return [row for cell_value, row in cell_row_pairs if cell_value < filter_value]

    def filter_date_lesser_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs: List[Tuple[Date, Row]] = []
        for row in rows:
            cell_value = row.values[column.name]
            if isinstance(cell_value, Date):
                cell_row_pairs.append((cell_value, row))
        return [row for cell_value, row in cell_row_pairs if cell_value <= filter_value]

    def filter_date_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs: List[Tuple[Date, Row]] = []
        for row in rows:
            cell_value = row.values[column.name]
            if isinstance(cell_value, Date):
                cell_row_pairs.append((cell_value, row))
        return [row for cell_value, row in cell_row_pairs if cell_value == filter_value]

    def filter_date_not_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs: List[Tuple[Date, Row]] = []
        for row in rows:
            cell_value = row.values[column.name]
            if isinstance(cell_value, Date):
                cell_row_pairs.append((cell_value, row))
        return [row for cell_value, row in cell_row_pairs if cell_value != filter_value]

    # These two are similar to the filter methods above, but operate on strings obtained from the
    # question, instead of dates or numbers.  So they check for whether the string value is present
    # in the cell or not, instead of using a numerical / date comparator.  We only add them to the
    # language if we see a string column in the table (which is basically always).

    def filter_in(self, rows: List[Row], column: StringColumn, filter_values: List[str]) -> List[Row]:
        # We accept a list of filter values instead of a single string to allow the outputs of select like
        # operations to be passed in as filter values.
        # Assuming filter value has underscores for spaces. The cell values also have underscores
        # for spaces, so we do not need to replace them here.
        # Note that if a list of filter values is passed, we only use the first one.
        if not filter_values:
            raise ExecutionError(f"Unexpected filter value: {filter_values}")
        if isinstance(filter_values, str):
            filter_value = filter_values
        elif isinstance(filter_values, list):
            filter_value = filter_values[0]
        else:
            raise ExecutionError(f"Unexpected filter value: {filter_values}")
        # Also, we need to remove the "string:" that was prepended to the entity name in the language.
        filter_value = filter_value.lstrip('string:')
        filtered_rows: List[Row] = []
        for row in rows:
            cell_value = row.values[column.name]
            if isinstance(cell_value, str) and filter_value in cell_value:
                filtered_rows.append(row)
        return filtered_rows

    def filter_not_in(self, rows: List[Row], column: StringColumn, filter_values: List[str]) -> List[Row]:
        # We accept a list of filter values instead of a single string to allow the outputs of select like
        # operations to be passed in as filter values.
        # Assuming filter value has underscores for spaces. The cell values also have underscores
        # for spaces, so we do not need to replace them here.
        # Note that if a list of filter values is passed, we only use the first one.
        if not filter_values:
            raise ExecutionError(f"Unexpected filter value: {filter_values}")
        if isinstance(filter_values, str):
            filter_value = filter_values
        elif isinstance(filter_values, list):
            filter_value = filter_values[0]
        else:
            raise ExecutionError(f"Unexpected filter value: {filter_values}")
        # Also, we need to remove the "string:" that was prepended to the entity name in the language.
        filter_value = filter_value.lstrip('string:')
        filtered_rows: List[Row] = []
        for row in rows:
            cell_value = row.values[column.name]
            if isinstance(cell_value, str) and filter_value not in cell_value:
                filtered_rows.append(row)
        return filtered_rows

    # These are some more date-column-specific functions, which only get added if we see a number
    # column.
    def max_date(self, rows: List[Row], column: DateColumn) -> Date:
        """
        Takes a list of rows and a column and returns the max of the values under that column in
        those rows.
        """
        cell_values = [row.values[column.name] for row in rows if row.values[column.name] is not None]
        if not cell_values:
            return Date(-1, -1, -1)
        if not all([isinstance(value, Date) for value in cell_values]):
            raise ExecutionError(f"Invalid values for date selection function: {cell_values}")
        return max(cell_values)  # type: ignore

    def min_date(self, rows: List[Row], column: DateColumn) -> Date:
        """
        Takes a list of rows and a column and returns the min of the values under that column in
        those rows.
        """
        cell_values = [row.values[column.name] for row in rows if row.values[column.name] is not None]
        if not cell_values:
            return Date(-1, -1, -1)
        if not all([isinstance(value, Date) for value in cell_values]):
            raise ExecutionError(f"Invalid values for date selection function: {cell_values}")
        return min(cell_values)  # type: ignore

    # These are some more number-column-specific functions, which only get added if we see a number
    # column.

    def max_number(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the max of the values under that column in
        those rows.
        """
        cell_values = [row.values[column.name] for row in rows if row.values[column.name] is not None]
        if not cell_values:
            return 0.0  # type: ignore
        if not all([isinstance(value, Number) for value in cell_values]):
            raise ExecutionError(f"Invalid values for number selection function: {cell_values}")
        return max(cell_values)  # type: ignore

    def min_number(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the min of the values under that column in
        those rows.
        """
        cell_values = [row.values[column.name] for row in rows if row.values[column.name] is not None]
        if not cell_values:
            return 0.0  # type: ignore
        if not all([isinstance(value, Number) for value in cell_values]):
            raise ExecutionError(f"Invalid values for number selection function: {cell_values}")
        return min(cell_values)  # type: ignore

    def sum(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the sum of the values under that column in
        those rows.
        """
        cell_values = [row.values[column.name] for row in rows if row.values[column.name] is not None]
        if not cell_values:
            return 0.0  # type: ignore
        return sum(cell_values)  # type: ignore

    def average(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the mean of the values under that column in
        those rows.
        """
        cell_values = [row.values[column.name] for row in rows if row.values[column.name] is not None]
        if not cell_values:
            return 0.0  # type: ignore
        return sum(cell_values) / len(cell_values)  # type: ignore

    def diff(self, first_row: List[Row], second_row: List[Row], column: NumberColumn) -> Number:
        """
        Takes a two rows and a number column and returns the difference between the values under
        that column in those two rows.
        """
        if not first_row or not second_row:
            return 0.0  # type: ignore
        first_value = first_row[0].values[column.name]
        second_value = second_row[0].values[column.name]
        if isinstance(first_value, float) and isinstance(second_value, float):
            return first_value - second_value  # type: ignore
        elif first_value is None or second_value is None:
            return 0.0  # type: ignore
        else:
            raise ExecutionError(f"Invalid column for diff: {column.name}")

    # End of language predicates.  Stuff below here is for private use, helping to implement the
    # functions above.

    def __eq__(self, other):
        if not isinstance(other, WikiTablesLanguage):
            return False
        return self.table_data == other.table_data

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

    def _get_most_frequent_values(self, rows: List[Row], column: Column) -> List[Any]:
        value_frequencies: Dict[CellValueType, int] = defaultdict(int)
        max_frequency = 0
        most_frequent_list: List[CellValueType] = []
        for row in rows:
            cell_value = row.values[column.name]
            if cell_value is not None:
                value_frequencies[cell_value] += 1
                frequency = value_frequencies[cell_value]
                if frequency > max_frequency:
                    max_frequency = frequency
                    most_frequent_list = [cell_value]
                elif frequency == max_frequency:
                    most_frequent_list.append(cell_value)
        return most_frequent_list
