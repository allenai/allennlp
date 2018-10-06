from typing import List, Tuple, Dict
from copy import deepcopy
from sqlite3 import Cursor

from parsimonious import Grammar

from allennlp.semparse.contexts.sql_context_utils import SqlVisitor
from allennlp.semparse.contexts.sql_context_utils import format_grammar_string, initialize_valid_actions
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import read_dataset_schema
from allennlp.semparse.contexts.text2sql_table_context import GRAMMAR_DICTIONARY
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_table_values
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_tables

class Text2SqlWorld:
    """
    World representation for any of the Text2Sql datasets.

    Parameters
    ----------
    schema_path: ``str``
        A path to a schema file which we read into a dictionary
        representing the SQL tables in the dataset, the keys are the
        names of the tables that map to lists of the table's column names.
    """
    def __init__(self,
                 schema_path: str,
                 cursor: Cursor = None) -> None:
        # NOTE: This base dictionary should not be modified.

        self.cursor = cursor
        self.schema = read_dataset_schema(schema_path)
        self.base_grammar_dictionary = self._initialize_grammar_dictionary(deepcopy(GRAMMAR_DICTIONARY))

    def get_action_sequence_and_all_actions(self,
                                            query: List[str],
                                            prelinked_entities: Dict[str, str] = None) -> Tuple[List[str], List[str]]: # pylint: disable=line-too-long
        grammar_with_context = deepcopy(self.base_grammar_dictionary)

        prelinked_entities = prelinked_entities or {}
        for token in prelinked_entities.keys():
            grammar_with_context["value"] = [f'"\'{token}\'"'] + grammar_with_context["value"]

        grammar = Grammar(format_grammar_string(grammar_with_context))

        valid_actions = initialize_valid_actions(grammar)
        all_actions = set()
        for action_list in valid_actions.values():
            all_actions.update(action_list)
        sorted_actions = sorted(all_actions)

        sql_visitor = SqlVisitor(grammar)
        action_sequence = sql_visitor.parse(" ".join(query)) if query else []
        return action_sequence, sorted_actions

    def _initialize_grammar_dictionary(self, grammar_dictionary: Dict[str, List[str]]) -> Dict[str, List[str]]:
        # Add all the table and column names to the grammar.
        if self.schema:
            update_grammar_with_tables(grammar_dictionary, self.schema)

            if self.cursor is not None:
                # Now if we have strings in the table, we need to be able to
                # produce them, so we find all of the strings in the tables here
                # and create production rules from them.
                grammar_dictionary["number"] = []
                grammar_dictionary["string"] = []

                update_grammar_with_table_values(grammar_dictionary, self.schema, self.cursor)

        return grammar_dictionary



    @staticmethod
    def is_global_rule(nonterminal: str) -> bool: # pylint: disable=unused-argument
        return True
