from collections import defaultdict

from typing import Callable, Dict, List, Set
import re

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.semparse.worlds.world import ParsingError, World
from allennlp.semparse.type_declarations import atis_type_declaration as types
from allennlp.semparse.contexts import atis_tables

from parsimonious.expressions import Sequence, OneOf
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from parsimonious.nodes import RegexNode

class AtisWorld(World):
    def __init__(self, conversation_context, utterance=None) -> None:
        self.conversation_context = conversation_context 
        self.utterance = utterance
        self.valid_actions = self.get_all_valid_actions()
        self.grammar_str = self.get_grammar_str()

    def get_all_valid_actions(self):
        valid_actions = self.conversation_context.valid_actions
        for local_str in self.get_local_strs():
            if local_str not in valid_actions['string']:
                valid_actions['string'].append(local_str)

        for local_num in atis_tables.get_nums_from_utterance(self.utterance):
            if local_num not in valid_actions['number']:
                valid_actions['number'].append(local_num)
        
        print(valid_actions)
        return valid_actions

    def get_grammar_str(self) -> str:
        grammar_str_with_context = self.conversation_context.base_sql_def 
        grammar_str_with_context += "\n    col_ref \t\t = " +" / ".join(self.valid_actions["col_ref"]) + " / asterisk"
        grammar_str_with_context += self.generate_one_of_str("table_name", sorted(self.valid_actions["table_name"], reverse=True))
        grammar_str_with_context += self.generate_one_of_str("number", sorted(self.valid_actions['number'], reverse=True))
        grammar_str_with_context += self.generate_one_of_str("string", ["'{}'".format(local_str) for local_str in self.valid_actions['string']])
        return grammar_str_with_context

    def get_local_strs(self) -> List[str]:
        local_strs: List[str] = []
        for city in atis_tables.CITIES:
            if city.lower() in self.utterance.lower():
                local_strs.append(city)

        for code in atis_tables.AIRPORT_CODES:
            if code.lower() in self.utterance.lower():
                local_strs.append(code)

        for state in atis_tables.STATES:
            if state.lower() in self.utterance.lower():
                local_strs.append(state)

        for state_codes in atis_tables.STATE_CODES:
            if state_codes.lower() in self.utterance.lower():
                local_strs.append(state_codes)

        for airline in atis_tables.AIRLINE_CODES.keys():
            if airline.lower() in self.utterance.lower() or atis_tables.AIRLINE_CODES[airline] in self.utterance.lower():
                local_strs.append(atis_tables.AIRLINE_CODES[airline])

        for service in atis_tables.GROUND_SERVICE.keys():
            if service.lower() in self.utterance.lower():
                local_strs.append(atis_tables.GROUND_SERVICE[service])

        return local_strs

    def generate_one_of_str(self, nonterminal: str, literals: List[str]) -> str:
        return  "\n{} \t\t = ".format(nonterminal) + " / ".join(['"{}"'.format(lit) for lit in literals])
    
    @overrides
    def get_valid_actions(self) -> Dict[str, List[str]]:
        valid_actions: Dict[str, List[str]] = defaultdict(set)
        for key in self.grammar:
            rhs = self.grammar[key]
            if isinstance(rhs, Sequence):
                valid_actions[key].add("{} -> {}".format(key, " ".join(rhs._unicode_members())))

        valid_action_strings = {key: sorted(value) for key, value in valid_actions.items()}
        return valid_action_strings

    def get_action_sequence(self, query: str) -> List[str]:
        grammar_with_context = Grammar(self.get_grammar_str())
        sql_visitor = SQLVisitor(grammar_with_context)
        query = query.split("\n")[0]
        if query:
            action_sequence = sql_visitor.parse(query)
            return action_sequence 

        return []


class SQLVisitor(NodeVisitor):

    def __init__(self, grammar):
        self.prod_acc = []
        self.grammar = grammar

        for nonterm in self.grammar.keys():
            if nonterm != 'stmt':
                self.__setattr__('visit_' + nonterm, self.add_prod_rule)

    def generic_visit(self, node, visited_children):
        self.add_prod_rule(node)

    def add_prod_rule(self, node, children=None):
        if node.expr.name and node.expr.name != 'ws':
            rule = '{} ='.format(node.expr.name)

            if isinstance(node, RegexNode):
                rule += '"{}"'.format(node.text)

            for child in node.__iter__():
                if child.expr.name != '':
                    rule += ' {}'.format(child.expr.name)
                else:
                    rule += ' {}'.format(child.expr._as_rhs())

            self.prod_acc = [rule] + self.prod_acc

    def visit_stmt(self, node, children):
        self.add_prod_rule(node)
        return self.prod_acc
         
