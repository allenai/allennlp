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
        
        trigger_lists = [atis_tables.CITIES,  atis_tables.AIRPORT_CODES,  atis_tables.STATES,  atis_tables.STATE_CODES,  
                       atis_tables.FARE_BASIS_CODE,  atis_tables.CLASS,  atis_tables.AIRLINE_CODE_LIST,  atis_tables.DAY_OF_WEEK]


        for trigger_list in trigger_lists:
            for trigger in trigger_list:
                if trigger.lower() in self.utterance.lower():
                    local_strs.append(trigger)
        
        trigger_dict_list = [atis_tables.AIRLINE_CODES, atis_tables.GROUND_SERVICE, atis_tables.YES_NO]

        for trigger_dict in trigger_dict_list:
            for trigger in trigger_dict.keys():
                if trigger.lower() in self.utterance.lower():
                    local_strs.append(trigger_dict[trigger])

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
    
    @overrides
    def all_possible_actions(self) -> List[str]:
        all_actions = set()
        for non_term, rhs_list in self.get_all_valid_actions().items():
            for rhs in rhs_list:
                if non_term == 'string':
                    all_actions.add('{} -> "\'{}\'"'.format(non_term, rhs))
                elif non_term == 'number' or non_term == 'table_name':
                    all_actions.add('{} -> "{}"'.format(non_term, rhs))

                else:
                    all_actions.add("{} -> {}".format(non_term, rhs))
        for opt in atis_tables.OPTIONAL:
            all_actions.add("{} ->".format(opt))
        return sorted(all_actions)


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
            rule = '{} ->'.format(node.expr.name)

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
         
