from typing import List, Tuple, Dict
from copy import deepcopy
from sqlite3 import Cursor
import os

from parsimonious import Grammar
from parsimonious.exceptions import ParseError

from allennlp.common.checks import ConfigurationError
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor
from allennlp.semparse.contexts.sql_context_utils import format_grammar_string, initialize_valid_actions
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import read_dataset_schema
from allennlp.semparse.contexts.text2sql_table_context import GRAMMAR_DICTIONARY
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_table_values
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_tables
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_global_values
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_to_be_variable_free
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_untyped_entities
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_values_with_variables
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_numbers_and_strings_with_variables

class Text2SqlWorld:
    """
    World representation for any of the Text2Sql datasets.

    Parameters
    ----------
    schema_path: ``str``
        A path to a schema file which we read into a dictionary
        representing the SQL tables in the dataset, the keys are the
        names of the tables that map to lists of the table's column names.
    cursor : ``Cursor``, optional (default = None)
        An optional cursor for a database, which is used to add
        database values to the grammar.
    use_prelinked_entities : ``bool``, (default = True)
        Whether or not to use the pre-linked entities from the text2sql data.
        We take this parameter here because it effects whether we need to add
        table values to the grammar.
    variable_free : ``bool``, optional (default = True)
        Denotes whether the data being parsed by the grammar is variable free.
        If it is, the grammar is modified to be less expressive by removing
        elements which are not necessary if the data is variable free.
    use_untyped_entities : ``bool``, optional (default = False)
        Whether or not to try to infer the types of prelinked variables.
        If not, they are added as untyped values to the grammar instead.
    """
    def __init__(self,
                 schema_path: str,
                 cursor: Cursor = None,
                 use_prelinked_entities: bool = True,
                 variable_free: bool = True,
                 use_untyped_entities: bool = False) -> None:
        self.cursor = cursor
        self.schema = read_dataset_schema(schema_path)
        self.columns = {column.name: column for table in self.schema.values() for column in table}
        self.dataset_name = os.path.basename(schema_path).split("-")[0]
        self.use_prelinked_entities = use_prelinked_entities
        self.variable_free = variable_free
        self.use_untyped_entities = use_untyped_entities

        # NOTE: This base dictionary should not be modified.
        self.base_grammar_dictionary = self._initialize_grammar_dictionary(deepcopy(GRAMMAR_DICTIONARY))

    def get_action_sequence_and_all_actions(self,
                                            query: List[str] = None,
                                            prelinked_entities: Dict[str, Dict[str, str]] = None) -> Tuple[List[str], List[str]]: # pylint: disable=line-too-long
        grammar_with_context = deepcopy(self.base_grammar_dictionary)

        if not self.use_prelinked_entities and prelinked_entities is not None:
            raise ConfigurationError("The Text2SqlWorld was specified to not use prelinked "
                                     "entities, but prelinked entities were passed.")
        prelinked_entities = prelinked_entities or {}

        if self.use_untyped_entities:
            update_grammar_values_with_variables(grammar_with_context, prelinked_entities)
        else:
            update_grammar_numbers_and_strings_with_variables(grammar_with_context,
                                                              prelinked_entities,
                                                              self.columns)

        grammar = Grammar(format_grammar_string(grammar_with_context))

        valid_actions = initialize_valid_actions(grammar)
        all_actions = set()
        for action_list in valid_actions.values():
            all_actions.update(action_list)
        sorted_actions = sorted(all_actions)

        sql_visitor = SqlVisitor(grammar)
        try:
            action_sequence = sql_visitor.parse(" ".join(query)) if query else []
        except ParseError:
            action_sequence = None

        return action_sequence, sorted_actions

    def _initialize_grammar_dictionary(self, grammar_dictionary: Dict[str, List[str]]) -> Dict[str, List[str]]:
        # Add all the table and column names to the grammar.
        update_grammar_with_tables(grammar_dictionary, self.schema)

        if self.cursor is not None and not self.use_prelinked_entities:
            # Now if we have strings in the table, we need to be able to
            # produce them, so we find all of the strings in the tables here
            # and create production rules from them. We only do this if
            # we haven't pre-linked entities, because if we have, we don't
            # need to be able to generate the values - just the placeholder
            # symbols which link to them.
            grammar_dictionary["number"] = []
            grammar_dictionary["string"] = []
            update_grammar_with_table_values(grammar_dictionary, self.schema, self.cursor)

        # Finally, update the grammar with global, non-variable values
        # found in the dataset, if present.
        update_grammar_with_global_values(grammar_dictionary, self.dataset_name)

        if self.variable_free:
            update_grammar_to_be_variable_free(grammar_dictionary)

        if self.use_untyped_entities:
            update_grammar_with_untyped_entities(grammar_dictionary)

        return grammar_dictionary

    def is_global_rule(self, production_rule: str) -> bool:
        if self.use_prelinked_entities:
            # we are checking -4 as is not a global rule if we
            # see the 0 in the a rule like 'value -> ["\'city_name0\'"]'
            if "value" in production_rule and production_rule[-4].isnumeric():
                return False
        return True
