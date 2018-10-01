from typing import List, Tuple
from copy import deepcopy

from parsimonious import Grammar

from allennlp.semparse.contexts.sql_context_utils import SqlTableContext, SqlVisitor
from allennlp.semparse.contexts.sql_context_utils import format_grammar_string, initialize_valid_actions


class Text2SqlWorld:
    """
    World representation for any of the Text2Sql datasets.

    Parameters
    ----------
    sql_table_context : ``SqlTableContext``
        This defines what sort of table-based constraints we apply
        to the world.
    """

    def __init__(self, sql_table_context: SqlTableContext) -> None:
        # NOTE: This base dictionary should not be modified.
        self.sql_table_context = sql_table_context
        self.base_grammar_dictionary = sql_table_context.get_grammar_dictionary()

    def get_action_sequence_and_all_actions(self, query: List[str]) -> Tuple[List[str], List[str]]:
        # TODO(Mark): Add in modifications here
        grammar_with_context = deepcopy(self.base_grammar_dictionary)
        grammar = Grammar(format_grammar_string(grammar_with_context))

        valid_actions = initialize_valid_actions(grammar)
        all_actions = set()
        for action_list in valid_actions.values():
            all_actions.update(action_list)
        all_actions = sorted(all_actions)

        sql_visitor = SqlVisitor(grammar)
        print(query)
        action_sequence = sql_visitor.parse(" ".join(query)) if query else []
        return action_sequence, all_actions

    @staticmethod
    def is_global_rule(nonterminal: str) -> bool:
        return True
