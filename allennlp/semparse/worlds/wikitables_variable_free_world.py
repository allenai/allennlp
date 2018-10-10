"""
We store the information related to context sensitive execution of logical forms here.
We assume that the logical forms are written in the variable-free language described in the paper
'Memory Augmented Policy Optimization for Program Synthesis with Generalization' by Liang et al.
The language is the main difference between this class and `WikiTablesWorld`. Also, this class defines
an executor for the variable-free logical forms.
"""
# TODO(pradeep): Merge this class with the `WikiTablesWorld` class, and move all the
# language-specific functionality into type declarations.
from typing import Dict, List, Set, Union
import re
import logging

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.semparse.worlds.world import ParsingError, World
from allennlp.semparse.type_declarations import wikitables_variable_free as types
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.executors import WikiTablesVariableFreeExecutor

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
            types.ROW_FILTER_WITH_GENERIC_COLUMN: 2,
            types.ROW_FILTER_WITH_COMPARABLE_COLUMN: 2,
            types.ROW_NUM_OP: 2,
            types.ROW_FILTER_WITH_COLUMN_AND_NUMBER: 3,
            types.ROW_FILTER_WITH_COLUMN_AND_DATE: 3,
            types.ROW_FILTER_WITH_COLUMN_AND_STRING: 3,
            types.NUM_DIFF_WITH_COLUMN: 3,
            }

    def __init__(self, table_context: TableQuestionContext) -> None:
        super().__init__(constant_type_prefixes={"string": types.STRING_TYPE,
                                                 "num": types.NUMBER_TYPE},
                         global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                         global_name_mapping=types.COMMON_NAME_MAPPING)
        # TODO (pradeep): Do we need constant type prefixes?
        self.table_context = table_context

        self._executor = WikiTablesVariableFreeExecutor(self.table_context.table_data)

        # For every new column name seen, we update this counter to map it to a new NLTK name.
        self._column_counter = 0

        # Adding entities and numbers seen in questions to the mapping.
        self._question_entities, question_numbers = table_context.get_entities_from_question()
        self._question_numbers = [number for number, _ in question_numbers]
        for entity in self._question_entities:
            self._map_name(f"string:{entity}", keep_mapping=True)

        for number_in_question in self._question_numbers:
            self._map_name(f"num:{number_in_question}", keep_mapping=True)

        # Adding -1 to mapping because we need it for dates where not all three fields are
        # specified.
        self._map_name(f"num:-1", keep_mapping=True)

        # Keeps track of column name productions so that we can add them to the agenda.
        self._column_productions_for_agenda: Dict[str, str] = {}

        # Adding column names to the local name mapping.
        for column_name, column_type in table_context.column_types.items():
            self._map_name(f"{column_type}_column:{column_name}", keep_mapping=True)

        self.global_terminal_productions: Dict[str, str] = {}
        for predicate, mapped_name in self.global_name_mapping.items():
            if mapped_name in self.global_type_signatures:
                signature = self.global_type_signatures[mapped_name]
                self.global_terminal_productions[predicate] = f"{signature} -> {predicate}"

        # We don't need to recompute this ever; let's just compute it once and cache it.
        self._valid_actions: Dict[str, List[str]] = None

    @overrides
    def _get_curried_functions(self) -> Dict[Type, int]:
        return WikiTablesVariableFreeWorld.curried_functions

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return types.STARTING_TYPES

    def _translate_name_and_add_mapping(self, name: str) -> str:
        if "_column:" in name:
            # Column name
            translated_name = "C%d" % self._column_counter
            self._column_counter += 1
            if name.startswith("number_column:"):
                column_type = types.NUMBER_COLUMN_TYPE
            elif name.startswith("string_column:"):
                column_type = types.STRING_COLUMN_TYPE
            else:
                column_type = types.DATE_COLUMN_TYPE
            self._add_name_mapping(name, translated_name, column_type)
            self._column_productions_for_agenda[name] = f"{column_type} -> {name}"
        elif name.startswith("string:"):
            # We do not need to translate these names.
            original_name = name.replace("string:", "")
            translated_name = name
            self._add_name_mapping(original_name, translated_name, types.STRING_TYPE)
        elif name.startswith("num:"):
            # NLTK throws an error if it sees a "." in constants, which will most likely happen
            # within numbers as a decimal point. We're changing those to underscores.
            translated_name = name.replace(".", "_")
            if re.match("num:-[0-9_]+", translated_name):
                # The string is a negative number. This makes NLTK interpret this as a negated
                # expression and force its type to be TRUTH_VALUE (t).
                translated_name = translated_name.replace("-", "~")
            original_name = name.replace("num:", "")
            self._add_name_mapping(original_name, translated_name, types.NUMBER_TYPE)
        return translated_name

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        if name not in types.COMMON_NAME_MAPPING and name not in self.local_name_mapping:
            if not keep_mapping:
                raise ParsingError(f"Encountered un-mapped name: {name}")
            translated_name = self._translate_name_and_add_mapping(name)
        else:
            if name in types.COMMON_NAME_MAPPING:
                translated_name = types.COMMON_NAME_MAPPING[name]
            else:
                translated_name = self.local_name_mapping[name]
        return translated_name

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
            if token == ["last", "bottom"]:
                agenda_items.append("last")

        if "how many" in question:
            if "sum" not in agenda_items and "average" not in agenda_items:
                # The question probably just requires counting the rows. But this is not very
                # accurate. The question could also be asking for a value that is in the table.
                agenda_items.append("count")
        agenda = []
        # Adding productions from the global set.
        for agenda_item in set(agenda_items):
            agenda.append(self.global_terminal_productions[agenda_item])

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
            if entity not in tokens_in_column_names:
                agenda.append(f"{types.STRING_TYPE} -> {entity}")

        for number in self._question_numbers:
            # The reason we check for the presence of the number in the question again is because
            # some of these numbers are extracted from number words like month names and ordinals
            # like "first". On looking at some agenda outputs, I found that they hurt more than help
            # in the agenda.
            if f"_{number}_" in normalized_question:
                agenda.append(f"{types.NUMBER_TYPE} -> {number}")

        return agenda

    def execute(self, logical_form: str) -> Union[List[str], int]:
        return self._executor.execute(logical_form)
