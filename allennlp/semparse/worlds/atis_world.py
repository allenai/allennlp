from typing import List, Dict
import re

from parsimonious.grammar import Grammar
from parsimonious.nodes import Node, NodeVisitor
from parsimonious.expressions import Literal

from allennlp.semparse.contexts import atis_tables

def generate_one_of_str(nonterminal: str, literals: List[str]) -> str:
    return  "\n{} \t\t = ".format(nonterminal) + " / ".join(['"{}"'.format(lit) for lit in literals])

class AtisWorld():
    """
    World representation for the Atis SQL domain.
    """
    def __init__(self, conversation_context, utterance=None) -> None:
        self.conversation_context = conversation_context
        self.utterance: str = utterance
        self.valid_actions: Dict[str, List[str]] = self.init_all_valid_actions()
        self.grammar_str: str = self.get_grammar_str()

    def get_valid_actions(self) -> Dict[str, List[str]]:
        return self.valid_actions

    def init_all_valid_actions(self) -> Dict[str, List[str]]:
        """
        We initialize the world's valid actions with that of the context. This means that the strings
        and numbers that were valid earlier in the interaction are also valid. We then add new valid strings
        and numbers from the current utterance.
        """
        valid_actions = self.conversation_context.valid_actions
        for local_str in self.get_local_strs():
            if local_str not in valid_actions['string']:
                valid_actions['string'].append(local_str)

        for local_num in atis_tables.get_nums_from_utterance(self.utterance):
            if local_num not in valid_actions['number']:
                valid_actions['number'].append(local_num)

        return valid_actions

    def get_grammar_str(self) -> str:
        """
        Generate a string that can be used to instantiate a ``Grammar`` object. The string is a sequence of
        rules that define the grammar.
        """

        grammar_str_with_context = self.conversation_context.base_sql_def

        grammar_str_with_context += "\n    col_ref \t\t = " +  \
                                    " / ".join(sorted(self.valid_actions['col_ref'], reverse=True)) + \
                                    " / asterisk"

        grammar_str_with_context += generate_one_of_str("table_name",
                                                        sorted(self.valid_actions["table_name"],
                                                               reverse=True))
        grammar_str_with_context += generate_one_of_str("number",
                                                        sorted(self.valid_actions['number'],
                                                               reverse=True))
        grammar_str_with_context += generate_one_of_str("string",
                                                        ["'{}'".format(local_str)
                                                         for local_str in
                                                         self.valid_actions['string']])
        return grammar_str_with_context


    def get_local_strs(self) -> List[str]:
        """
        Based on the current utterance, return a list of valid strings and numbers that should be added.
        """
        local_strs: List[str] = []
        trigger_lists = [atis_tables.CITIES, atis_tables.AIRPORT_CODES,
                         atis_tables.STATES, atis_tables.STATE_CODES,
                         atis_tables.FARE_BASIS_CODE, atis_tables.CLASS,
                         atis_tables.AIRLINE_CODE_LIST, atis_tables.DAY_OF_WEEK,
                         atis_tables.CITY_CODE_LIST, atis_tables.MEALS,
                         atis_tables.RESTRICT_CODES]

        for trigger_list in trigger_lists:
            for trigger in trigger_list:
                if trigger.lower() in self.utterance.lower():
                    local_strs.append(trigger)

        trigger_dict_list = [atis_tables.AIRLINE_CODES,
                             atis_tables.CITY_CODES,
                             atis_tables.GROUND_SERVICE,
                             atis_tables.YES_NO,
                             atis_tables.MISC_STR]

        for trigger_dict in trigger_dict_list:
            for trigger in trigger_dict:
                if trigger.lower() in self.utterance.lower():
                    local_strs.append(trigger_dict[trigger])

        local_strs.extend(atis_tables.DAY_OF_WEEK)
        return local_strs

    def get_action_sequence(self, query: str) -> List[str]:
        grammar_with_context = Grammar(self.get_grammar_str())
        sql_visitor = SQLVisitor(grammar_with_context)
        query = query.split("\n")[0]
        if query:
            action_sequence = sql_visitor.parse(query)
            return action_sequence
        return []

    def all_possible_actions(self) -> List[str]:
        """
        Return a list of strings representing all possible actions
        of the form lhs -> [rhs]
        """
        all_actions = set()
        for non_term, rhs_list in self.valid_actions.items():
            for rhs in rhs_list:
                if non_term == 'string':
                    all_actions.add('{} -> ["\'{}\'"]'.format(non_term, rhs))

                elif non_term in ['number', 'asterisk', 'table_name']:
                    all_actions.add('{} -> ["{}"]'.format(non_term, rhs))

                else:
                    ws_str = rhs.lstrip("(").rstrip(")")
                    curr_child_strs = [tok for tok in re.split(" ws |ws | ws", ws_str) if tok]
                    all_actions.add("{} -> [{}]".format(non_term, ", ".join(curr_child_strs)))

        return sorted(all_actions)


class SQLVisitor(NodeVisitor):
    """
    ``SQLVisitor`` performs a depths-first traversal of the the AST. It takes the parse tree
    and gives us a action sequence that resulted in that parse.

    """
    def __init__(self, grammar: str) -> None:
        """
        Parameters
        __________
        grammar : ``str``
            A string that descrbies the PEG (parsing expression grammar) in the form of:
                nonterminal = ...
        """
        self.prod_acc: List[str] = []
        self.grammar: str = grammar

        for nonterm in self.grammar.keys():
            if nonterm != 'stmt':
                self.__setattr__('visit_' + nonterm, self.add_prod_rule)

    def generic_visit(self, node: Node, visited_children) -> None:
        self.add_prod_rule(node)

    def add_prod_rule(self, node: Node, children=None) -> None:
        """
        For each node, we accumulate the rules that generated its children in a list
        """
        if node.expr.name and node.expr.name != 'ws':
            lhs = '{} -> '.format(node.expr.name)

            if isinstance(node.expr, Literal):
                rhs = '["{}"]'.format(node.text)

            else:
                child_strs = []
                for child in node.__iter__():
                    if child.expr.name == 'ws':
                        continue
                    if child.expr.name != '':
                        child_strs.append(child.expr.name)
                    else:
                        ws_str = child.expr._as_rhs().lstrip("(").rstrip(")")
                        curr_child_strs = [tok for tok in re.split(" ws |ws | ws", ws_str) if tok]
                        child_strs.extend(curr_child_strs)
                rhs = "[" + ", ".join(child_strs) + "]"

            rule = lhs + rhs
            self.prod_acc = [rule] + self.prod_acc

    def visit_stmt(self, node: Node, children=None) -> List[str]:
        """
        The top level is a statement, so return the productions that we have accumulated
        """
        self.add_prod_rule(node)
        return self.prod_acc
