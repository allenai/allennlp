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

SQL_GRAMMAR_STR = r"""
    stmt                = query ";" ws

    query               = ws lparen?  ws "SELECT" ws "DISTINCT"? ws select_results ws "FROM" ws table_refs ws where_clause rparen?  ws
    select_results      = agg / col_refs

    agg                 = agg_func ws lparen ws col_ref ws rparen
    agg_func            = "MIN" / "min" / "MAX" / "max" / "COUNT" / "count"

    col_refs            = (col_ref (ws "," ws col_ref)*)

    table_refs          = table_name (ws "," ws table_name)*


    where_clause        = "WHERE" ws lparen? ws condition_paren (ws conj ws condition_paren)* ws rparen? ws

    condition_paren     = not? (lparen ws)? condition_paren2 (ws rparen)?
    condition_paren2    = not? (lparen ws)? condition_paren3 (ws rparen)?
    condition_paren3    = not? (lparen ws)? condition (ws rparen)?
    condition           = in_clause / ternaryexpr / biexpr

    in_clause           = (lparen ws)? col_ref ws "IN" ws query (ws rparen)?

    biexpr              = ( col_ref ws binaryop ws value) / (value ws binaryop ws value) / ( col_ref ws "LIKE" ws string)
    binaryop            = "+" / "-" / "*" / "/" / "=" /
                          ">=" / "<=" / ">" / "<"  / "is" / "IS"

    ternaryexpr         = col_ref ws not? "BETWEEN" ws value ws and value ws

    value               = not? ws? pos_value
    pos_value           = ("ALL" ws query) / ("ANY" ws query) / number / boolean / col_ref / string / agg_results / "NULL"

    agg_results         = ws lparen?  ws "SELECT" ws "DISTINCT"? ws agg ws "FROM" ws table_name ws where_clause rparen?  ws

    boolean             = "true" / "false"

    ws                  = ~"\s*"i

    lparen              = "("
    rparen              = ")"
    conj                = and / or
    and                 = "AND" ws
    or                  = "OR" ws
    not                 = ("NOT" ws ) / ("not" ws)
    asterisk            = "*"

"""

class AtisWorld(World):
    def __init__(self, utterance=None) -> None:
        super(AtisWorld, self).__init__(constant_type_prefixes={"string": types.STRING_TYPE,
                                                                "num": types.NUM_TYPE,
                                                                "ent": types.ENTITY_TYPE},
                                        global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                                        global_name_mapping=types.COMMON_NAME_MAPPING)

        self.utterance = utterance
        self.grammar = Grammar(self.get_grammar_str_with_context())

    curried_functions = {
        types.CONJ_TYPE: 2,
        types.BINOP_TYPE: 2,
        types.SELECT_TYPE: 3,
        types.FROM_TYPE: 1,
        types.WHERE_TYPE: 1,
        types.IN_TYPE: 2
        }


    def _get_curried_functions(self) -> Dict[Type, int]:
        return AtisWorld.curried_functions

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        return types.COMMON_NAME_MAPPING[name] if name in types.COMMON_NAME_MAPPING else name

    @overrides
    def get_valid_actions(self) -> Dict[str, List[str]]:
        valid_actions: Dict[str, List[str]] = defaultdict(set)
        for key in self.grammar:
            rhs = self.grammar[key]
            if isinstance(rhs, Sequence):
                valid_actions[key].add("{} -> {}".format(key, " ".join(rhs._unicode_members())))

        # valid_actions = self.get_local_actions(valid_actions)

        valid_action_strings = {key: sorted(value) for key, value in valid_actions.items()}
        return valid_action_strings

    def get_local_actions(self, valid_actions):
        for city in atis_tables.CITIES:
            if city.lower() in self.utterance.lower():
                valid_actions["string"].add("{} -> {}".format("string", city))

        for airline in atis_tables.AIRLINE_CODES.keys():
            if airline.lower() in self.utterance.lower():
                valid_actions["string"].add("{} -> {}".format("string", atis_tables.AIRLINE_CODES[airline]))

        nums = atis_tables.get_nums_from_utterance(self.utterance) 
        for num in nums:
            valid_actions["number"].add("{} -> {}".format("number", num))

        return valid_actions

    def get_action_sequence(self, query: str) -> List[str]:
        grammar_with_context = Grammar(self.get_grammar_str_with_context())
        sql_visitor = SQLVisitor(grammar_with_context)
        query = query.strip("\n")
        action_sequence = sql_visitor.parse(query)
        return action_sequence 

    def get_local_strs(self) -> List[str]:
        local_strs: List[str] = []
        for city in atis_tables.CITIES:
            if city.lower() in self.utterance.lower():
                local_strs.append(city)

        for airline in atis_tables.AIRLINE_CODES.keys():
            if airline.lower() in self.utterance.lower():
                local_strs.append(city)
        return local_strs

    def get_grammar_str_with_context(self) -> str:
        grammar_str_with_context = SQL_GRAMMAR_STR
        grammar_str_with_context  += "\n    col_ref \t\t = " 
        
        table_col_pairs = []

        for table in atis_tables.TABLES.keys():
            for column in sorted(atis_tables.TABLES[table], reverse=True):
                table_col_pairs.append('("{}" ws "." ws "{}")'.format(table, column))
        
        grammar_str_with_context += " / ".join(table_col_pairs) + " / asterisk"
        grammar_str_with_context += "\n    table_name \t\t = " + " / ".join(['"{}"'.format(table) for table in list(sorted(atis_tables.TABLES.keys(), reverse=True))])
        
        nums = atis_tables.get_nums_from_utterance(self.utterance)
        grammar_str_with_context += "\n    number \t\t = " + " / ".join(['"{}"'.format(num) for num in nums])

        local_strs = self.get_local_strs()
        grammar_str_with_context += "\n    string \t\t = " + " / ".join(['''"'{}'"'''.format(local_str) for local_str in local_strs])

        return grammar_str_with_context


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
         

