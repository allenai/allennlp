"""
We store the information related to context sensitive execution of logical forms here.
We assume that the logical forms are written in the variable-free language described in the paper
'Memory Augmented Policy Optimization for Program Synthesis with Generalization' by Liang et al.
The language is the main difference between this class and `WikiTablesWorld`. Also, this class defines
an executor for the variable-free logical forms.
"""
# TODO(pradeep): Merge this class with the `WikiTablesWorld` class, and move all the
# language-specific functionality into type declarations.
from typing import Dict, List, Set, Union, Tuple
import re
import logging

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.semparse import util as semparse_util
from allennlp.semparse.worlds.world import ParsingError, ExecutionError, World
from allennlp.semparse.type_declarations import wikitables_variable_free as types
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class WikiTablesVariableFreeWorld(World):
    """
    World representation for the WikitableQuestions domain with the variable-free language used in
    the paper from Liang et al. (2018).

    Parameters
    ----------
    table_graph : ``TableQuestionKnowledgeGraph``
        Context associated with this world.
    """
    # When we're converting from logical forms to action sequences, this set tells us which
    # functions in the logical form are curried functions, and how many arguments the function
    # actually takes.  This is necessary because NLTK curries all multi-argument functions to a
    # series of one-argument function applications.  See `world._get_transitions` for more info.
    curried_functions = {
            types.SELECT_TYPE: 2,
            types.ROW_FILTER_WITH_COLUMN: 2,
            types.ROW_NUM_OP: 2,
            types.ROW_FILTER_WITH_COLUMN_AND_NUMBER: 3,
            types.ROW_FILTER_WITH_COLUMN_AND_DATE: 3,
            types.ROW_FILTER_WITH_COLUMN_AND_STRING: 3,
            types.NUM_DIFF_WITH_COLUMN: 3,
            }

    def __init__(self, table_graph: TableQuestionKnowledgeGraph) -> None:
        super().__init__(constant_type_prefixes={"string": types.STRING_TYPE,
                                                 "num": types.NUMBER_TYPE},
                         global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                         global_name_mapping=types.COMMON_NAME_MAPPING)
        self.table_graph = table_graph

        # For every new Sempre column name seen, we update this counter to map it to a new NLTK name.
        self._column_counter = 0

        # This adds all of the cell and column names to our local name mapping.
        for entity in table_graph.entities:
            self._map_name(entity, keep_mapping=True)

        self._entity_set = set(table_graph.entities)
        self.terminal_productions: Dict[str, str] = {}
        for entity in self._entity_set:
            mapped_name = self.local_name_mapping[entity]
            signature = self.local_type_signatures[mapped_name]
            self.terminal_productions[entity] = f"{signature} -> {entity}"

        for predicate, mapped_name in self.global_name_mapping.items():
            if mapped_name in self.global_type_signatures:
                signature = self.global_type_signatures[mapped_name]
                self.terminal_productions[predicate] = f"{signature} -> {predicate}"

        # We don't need to recompute this ever; let's just compute it once and cache it.
        self._valid_actions: Dict[str, List[str]] = None

    def is_table_entity(self, entity_name: str) -> bool:
        """
        Returns ``True`` if the given entity is one of the entities in the table.
        """
        return entity_name in self._entity_set

    @overrides
    def _get_curried_functions(self) -> Dict[Type, int]:
        return WikiTablesVariableFreeWorld.curried_functions

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return types.STARTING_TYPES

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        if name not in types.COMMON_NAME_MAPPING and name not in self.local_name_mapping:
            if not keep_mapping:
                raise ParsingError(f"Encountered un-mapped name: {name}")
            if name.startswith("fb:row.row"):
                # Column name
                translated_name = "C%d" % self._column_counter
                self._column_counter += 1
                self._add_name_mapping(name, translated_name, types.COLUMN_TYPE)
            elif name.startswith("fb:cell"):
                # Cell name
                translated_name = "string:%s" % name.split(".")[-1]
                self._add_name_mapping(name, translated_name, types.STRING_TYPE)
            elif name.startswith("fb:part"):
                # part name
                translated_name = "string:%s" % name.split(".")[-1]
                self._add_name_mapping(name, translated_name, types.STRING_TYPE)
            else:
                # The only other unmapped names we should see are numbers.
                # NLTK throws an error if it sees a "." in constants, which will most likely happen
                # within numbers as a decimal point. We're changing those to underscores.
                translated_name = name.replace(".", "_")
                if re.match("-[0-9_]+", translated_name):
                    # The string is a negative number. This makes NLTK interpret this as a negated
                    # expression and force its type to be TRUTH_VALUE (t).
                    translated_name = translated_name.replace("-", "~")
                translated_name = f"num:{translated_name}"
                self._add_name_mapping(name, translated_name, types.NUMBER_TYPE)
        else:
            if name in types.COMMON_NAME_MAPPING:
                translated_name = types.COMMON_NAME_MAPPING[name]
            else:
                translated_name = self.local_name_mapping[name]
        return translated_name

    def get_agenda(self):
        agenda_items = self.table_graph.get_linked_agenda_items()
        # Global rules
        question_tokens = [token.text for token in self.table_graph.question_tokens]
        question = " ".join(question_tokens)
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
            if token in ["least", "top", "first", "smallest", "shortest", "lowest"]:
                # This condition is too brittle. But for most logical forms with "min", there are
                # semantically equivalent ones with "argmin". The exceptions are rare.
                if "what is the least" in question:
                    agenda_items.append("min")
                else:
                    agenda_items.append("argmin")
            if token in ["last", "most", "largest", "highest", "longest", "greatest"]:
                # This condition is too brittle. But for most logical forms with "max", there are
                # semantically equivalent ones with "argmax". The exceptions are rare.
                if "what is the most" in question:
                    agenda_items.append("max")
                else:
                    agenda_items.append("argmax")

        if "how many" in question or "number" in question:
            if "sum" not in agenda_items and "average" not in agenda_items:
                # The question probably just requires counting the rows. But this is not very
                # accurate. The question could also be asking for a value that is in the table.
                agenda_items.append("count")
        agenda = []
        for agenda_item in set(agenda_items):
            agenda.append(self.terminal_productions[agenda_item])
        return agenda

    def execute(self, logical_form: str) -> Union[List[str], int]:
        if not logical_form.startswith("("):
            logical_form = f"({logical_form})"
        logical_form = logical_form.replace(",", " ")
        expression_as_list = semparse_util.lisp_to_nested_expression(logical_form)
        # Expression list has an additional level of nesting at the top. For example, if the logical
        # for is "(select all_rows fb:row.row.league)", the expression list will be
        # [['select', 'all_rows', 'fb:row.row.league']].
        # Removing the top most level of nesting.
        result = self.handle_expression(expression_as_list[0])
        return result

    def handle_expression(self, expression_list: Union[List[str], str]):
        if isinstance(expression_list, list):
            # This is a function application.
            function_name = expression_list[0]
            expression_is_application = True
        else:
            # This is a constant (likst "all_rows")
            function_name = expression_list
            expression_is_application = False
        try:
            function = getattr(self, f"_{function_name}")
            return function(expression_list[1:]) if expression_is_application else function()
        except AttributeError:
            logger.error("Function not found: %s", function_name)
            raise ExecutionError(f"Function not found: {function_name}")

    def _get_row_list_and_column_name(self, expression_list: List[str]) -> Tuple[List[Dict[str,
                                                                                           str]],
                                                                                 str]:
        """
        Utility function for computing the initial row list and a column name from an expression for
        all functions that need these operations, like "select", "argmax", "argmin", etc.
        """
        row_list: List[Dict[str, str]] = self.handle_expression(expression_list[0])
        column_name = expression_list[1]
        if not (isinstance(column_name, str) and column_name.startswith("fb:row.row.")):
            logger.error("Invalid column for selection: %s", column_name)
            raise ExecutionError(f"Invalid column for selection: {column_name}")
        if column_name not in row_list[0]:
            logger.error("Input list of rows do not contain column: %s", column_name)
            raise ExecutionError(f"Input list of rows do not contain column: {column_name}")
        return row_list, column_name

    def _all_rows(self) -> List[Dict[str, str]]:
        return self.table_graph.table_data

    def _select(self, expression_list: List[str]) -> List[str]:
        """
        Select function takes a list of rows and a column (decoded from the `expression_list`) and
        returns a list of cell values as strings.
        """
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        return [row[column_name] for row in row_list]

    def _argmax(self, expression_list: List[str]) -> List[Dict[str, str]]:
        """
        Takes a list of rows and a column (decoded from the `expression_list`) and returns a list
        containing a single row (dict from columns to cells) that has the maximum numerical value in
        the given column. We return a list instead of a single dict to be consistent with the return
        type of `_select` and `_all_rows`.
        """
        # TODO(pradeep): Deal with dates as well.
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        try:
            cell_row_pairs = [(float(row[column_name].replace('fb:cell.', '')), row) for row in row_list]
        except ValueError:
            # This means that at least one of the cells is not numerical.
            return []
        # Returns a list containing the row with the max cell value.
        return [sorted(cell_row_pairs, reverse=True)[0][1]]

    def _argmin(self, expression_list: List[str]) -> List[Dict[str, str]]:
        """
        Takes a list of rows and a column (decoded from the `expression_list`) and returns a list
        containing a single row (dict from columns to cells) that has the minimum numerical value in
        the given column. We return a list instead of a single dict to be consistent with the return
        type of `_select` and `_all_rows`.
        """
        # TODO(pradeep): Deal with dates as well.
        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        try:
            cell_row_pairs = [(float(row[column_name].replace('fb:cell.', '')), row) for row in row_list]
        except ValueError:
            # This means that at least one of the cells is not numerical.
            return []
        # Returns a list containing the row with the min cell value.
        return [sorted(cell_row_pairs)[0][1]]

    def _get_numbers_row_pairs_to_filter(self, expression_list: List[str]) -> Tuple[List[Tuple[float,
                                                                                               Dict[str, str]]],
                                                                                    float]:

        row_list, column_name = self._get_row_list_and_column_name(expression_list)
        if not row_list:
            return []
        try:
            # Note: This restricts the logical forms to only contain numbers as the second arguments
            # of filter functions. That is, we cannot evaluate something like
            # `(filter_* all_rows fb:row.row.number (count all_rows))`.
            filter_value = float(expression_list[2])
        except ValueError:
            logger.error("Invalid filter value: %s", expression_list[2])
            raise ExecutionError(f"Invalid filter value: {expression_list[2]}")
        try:
            cell_row_pairs = [(float(row[column_name].replace('fb:cell.', '')), row) for row in row_list]
        except ValueError:
            # This means that at least one of the cells is not numerical.
            return []
        return cell_row_pairs, filter_value

    def _filter_number_greater(self, expression_list: List[str]) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column is greater than the given value.
        """
        return_list = []
        cell_row_pairs, filter_value = self._get_numbers_row_pairs_to_filter(expression_list)
        for cell_value, row in cell_row_pairs:
            if cell_value > filter_value:
                return_list.append(row)
        return return_list

    def _filter_number_greater_equals(self, expression_list: List[str]) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column is greater than or equal to the given value.
        """
        return_list = []
        cell_row_pairs, filter_value = self._get_numbers_row_pairs_to_filter(expression_list)
        for cell_value, row in cell_row_pairs:
            if cell_value >= filter_value:
                return_list.append(row)
        return return_list

    def _filter_number_lesser(self, expression_list: List[str]) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column is lesser than the given value.
        """
        return_list = []
        cell_row_pairs, filter_value = self._get_numbers_row_pairs_to_filter(expression_list)
        for cell_value, row in cell_row_pairs:
            if cell_value < filter_value:
                return_list.append(row)
        return return_list

    def _filter_number_lesser_equals(self, expression_list: List[str]) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column is lesser than or equal to the given value.
        """
        return_list = []
        cell_row_pairs, filter_value = self._get_numbers_row_pairs_to_filter(expression_list)
        for cell_value, row in cell_row_pairs:
            if cell_value <= filter_value:
                return_list.append(row)
        return return_list

    def _filter_number_equals(self, expression_list: List[str]) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column equals the given value.
        """
        return_list = []
        cell_row_pairs, filter_value = self._get_numbers_row_pairs_to_filter(expression_list)
        for cell_value, row in cell_row_pairs:
            if cell_value == filter_value:
                return_list.append(row)
        return return_list

    def _filter_number_not_equals(self, expression_list: List[str]) -> List[Dict[str, str]]:
        """
        Takes a list of rows, a column, and a numerical value (decoded from `expression_list`) and
        returns all the rows where the value in that column is not equal to the given value.
        """
        return_list = []
        cell_row_pairs, filter_value = self._get_numbers_row_pairs_to_filter(expression_list)
        for cell_value, row in cell_row_pairs:
            if cell_value != filter_value:
                return_list.append(row)
        return return_list
