"""
A ``Text2SqlTableContext`` represents the SQL context in which an utterance appears
for the any of the text2sql datasets, with the grammar and the valid actions.
"""
from typing import List, Dict
import sqlite3
from copy import deepcopy


from parsimonious.grammar import Grammar

from allennlp.common.file_utils import cached_path
from allennlp.semparse.contexts.sql_context_utils import initialize_valid_actions

GRAMMAR_DICTIONARY = {}
GRAMMAR_DICTIONARY["statement"] = ['(query ws ";")', '(query ws)']
GRAMMAR_DICTIONARY["query"] = ['(ws select_core groupby_clause ws orderby_clause ws limit)',
                               '(ws select_core groupby_clause ws orderby_clause)',
                               '(ws select_core groupby_clause ws limit)',
                               '(ws select_core orderby_clause ws limit)',
                               '(ws select_core groupby_clause)',
                               '(ws select_core orderby_clause)',
                               '(ws select_core)']

GRAMMAR_DICTIONARY["select_core"] = ['(select_with_distinct select_results from_clause where_clause)',
                                     '(select_with_distinct select_results from_clause)',
                                     '(select_with_distinct select_results where_clause)',
                                     '(select_with_distinct select_results)']
GRAMMAR_DICTIONARY["select_with_distinct"] = ['(SELECT DISTINCT)', 'SELECT']
GRAMMAR_DICTIONARY["select_results"] = ['(ws select_result ws "," ws select_results)', '(ws select_result)']
GRAMMAR_DICTIONARY["select_result"] = ['sel_res_all_star', 'sel_res_tab_star', 'sel_res_val', 'sel_res_col']
GRAMMAR_DICTIONARY["sel_res_tab_star"] = ['name ".*"']
GRAMMAR_DICTIONARY["sel_res_all_star"] = ['"*"']
GRAMMAR_DICTIONARY['sel_res_val'] = ['(expr AS wsp name)', 'expr']
GRAMMAR_DICTIONARY['sel_res_col'] = ['col_ref AS wsp name']

GRAMMAR_DICTIONARY["from_clause"] = ['FROM source']
GRAMMAR_DICTIONARY["source"] = ['(ws single_source ws "," ws source)', '(ws single_source)']
GRAMMAR_DICTIONARY["single_source"] = ['source_table', 'source_subq']
GRAMMAR_DICTIONARY["source_table"] = ['table_name AS wsp name']
GRAMMAR_DICTIONARY["source_subq"] = ['("(" ws query ws ")" AS ws name)', '("(" ws query ws ")")']

GRAMMAR_DICTIONARY["where_clause"] = ['(WHERE wsp expr where_conj)', '(WHERE wsp expr)']
GRAMMAR_DICTIONARY["where_conj"] = ['(AND wsp expr where_conj)', '(AND wsp expr)']

GRAMMAR_DICTIONARY["groupby_clause"] = ['(GROUP BY group_clause having_clause)', '(GROUP BY group_clause)']
GRAMMAR_DICTIONARY["group_clause"] = ['(ws expr ws "," group_clause)', '(ws expr)']
GRAMMAR_DICTIONARY["having_clause"] = ['HAVING ws expr']

GRAMMAR_DICTIONARY["orderby_clause"] = ['ORDER BY order_clause']
GRAMMAR_DICTIONARY["order_clause"] = ['(ordering_term ws "," order_clause)', 'ordering_term']
GRAMMAR_DICTIONARY["ordering_term"] = ['(ws expr ordering)', 'ws expr']
GRAMMAR_DICTIONARY["ordering"] = ['ASC', 'DESC']
GRAMMAR_DICTIONARY["limit"] = ['LIMIT ws number']

GRAMMAR_DICTIONARY["col_ref"] = ['(table_name ws "." ws column_name)', 'column_name']
GRAMMAR_DICTIONARY["table_name"] = ['name']
GRAMMAR_DICTIONARY["column_name"] = ['name']
GRAMMAR_DICTIONARY["ws"] = ['~"\s*"i']
GRAMMAR_DICTIONARY['wsp'] = ['~"\s+"i']
GRAMMAR_DICTIONARY['name'] = ['~"[a-zA-Z]\w*"i']

GRAMMAR_DICTIONARY["expr"] = ['in_expr', 'like_expr', 'between_expr', 'binary_expr',
                              'unary_expr', 'null_check_expr', 'source_subq', 'value']
GRAMMAR_DICTIONARY["like_expr"] = ['value wsp LIKE ws string']
GRAMMAR_DICTIONARY["in_expr"] = ['(value wsp NOT IN wsp string_set)',
                                 '(value wsp IN wsp string_set)',
                                 '(value wsp NOT IN wsp expr)',
                                 '(value wsp IN wsp expr)']

GRAMMAR_DICTIONARY["between_expr"] = ['value BETWEEN wsp value AND wsp value']
GRAMMAR_DICTIONARY["binary_expr"] = ['value ws binaryop wsp expr']
GRAMMAR_DICTIONARY["unary_expr"] = ['unary_op expr']
GRAMMAR_DICTIONARY["null_check_expr"] = ['(col_ref IS NOT NULL)', '(col_ref IS NULL)']
GRAMMAR_DICTIONARY["value"] = ['parenval', 'datetime', 'number', 'boolean', 'function', 'col_ref', 'string']
GRAMMAR_DICTIONARY["datetime"] = ['"YEAR(CURDATE())"']
GRAMMAR_DICTIONARY["parenval"] = ['"(" ws expr ws ")"']
GRAMMAR_DICTIONARY["function"] = ['fname ws "(" ws DISTINCT ws arg_list_or_star ws ")")',
                                  '(fname ws "(" ws arg_list_or_star ws ")")']
GRAMMAR_DICTIONARY["arg_list_or_star"] = ['arg_list', '"*"']
GRAMMAR_DICTIONARY["arg_list"] = ['(expr ws "," ws arg_list)', 'expr']
GRAMMAR_DICTIONARY["number"] = ['~"\d*\.?\d+"i']
GRAMMAR_DICTIONARY["string_set"] = ['ws "(" ws string_set_vals ws ")"']
GRAMMAR_DICTIONARY["string_set_vals"] = ['(string ws "," ws string_set_vals)', 'string']
GRAMMAR_DICTIONARY["string"] = ['~"\'.*?\'"i']
GRAMMAR_DICTIONARY["fname"] = ['"COUNT"', '"SUM"', '"MAX"', '"MIN"', '"AVG"', '"ALL"']
GRAMMAR_DICTIONARY["boolean"] = ['"true"', '"false"']

# TODO(MARK): This is not tight enough. AND/OR are strictly boolean value operators.
GRAMMAR_DICTIONARY["binaryop"] = ['"+"', '"-"', '"*"', '"/"', '"="', '"<>"',
                                  '">="', '"<="', '">"', '"<"', 'AND', 'OR', 'LIKE']
GRAMMAR_DICTIONARY["unaryop"] = ['"+"', '"-"', '"not"', '"NOT"']

KEYWORDS = ["SELECT", "FROM", "WHERE", "AS", "LIKE", "AND", "OR", "DISTINCT", "GROUP",
            "ORDER", "BY", "ASC", "DESC", "BETWEEN", "IN", "IS", "NOT", "NULL", "HAVING", "LIMIT", "LIKE"]

for keyword in KEYWORDS:
    GRAMMAR_DICTIONARY[keyword] = f'"{keyword}"'

class Text2SqlTableContext:
    """
    A ``Text2SqlTableContext`` represents the SQL context with a grammar of SQL and the valid actions
    based on the schema of the tables that it represents.

    Parameters
    ----------
    all_tables: ``Dict[str, List[str]]``
        A dictionary representing the SQL tables in the dataset, the keys are the names of the tables
        that map to lists of the table's column names.
    tables_with_strings: ``Dict[str, List[str]]``
        A dictionary representing the SQL tables that we want to generate strings for. The keys are the
        names of the tables that map to lists of the table's column names.
    database_file : ``str``, optional
        The directory to find the sqlite database file. We query the sqlite database to find the strings
        that are allowed.
    """
    def __init__(self,
                 all_tables: Dict[str, List[str]] = None,
                 tables_with_strings: Dict[str, List[str]] = None,
                 database_file: str = None) -> None:
        self.grammar_dictionary = deepcopy(GRAMMAR_DICTIONARY)
        self.all_tables = all_tables
        self.tables_with_strings = tables_with_strings
        if database_file:
            self.database_file = cached_path(database_file)
            self.connection = sqlite3.connect(self.database_file)
            self.cursor = self.connection.cursor()

        self.grammar_str: str = self.initialize_grammar_str()
        self.grammar: Grammar = Grammar(self.grammar_str)
        self.valid_actions: Dict[str, List[str]] = initialize_valid_actions(self.grammar)
        if database_file:
            self.connection.close()

    def initialize_grammar_str(self):

        # Add all the tables names to the grammar.
        if self.all_tables:
            self.grammar_dictionary['table_name'] = \
                    sorted([f'"{table}"' for table in list(self.all_tables.keys())],
                           reverse=True)
            self.grammar_dictionary['col_ref'] = ['"*"']

            for table, columns in self.all_tables.items():
                self.grammar_dictionary['col_ref'].extend([f'("{table}" ws "." ws "{column}")'
                                                           for column in columns])
            self.grammar_dictionary['col_ref'] = sorted(self.grammar_dictionary['col_ref'], reverse=True)

        biexprs = []
        if self.tables_with_strings:
            for table, columns in self.tables_with_strings.items():
                biexprs.extend([f'("{table}" ws "." ws "{column}" ws binaryop ws {table}_{column}_string)'
                                for column in columns])
                for column in columns:
                    self.cursor.execute(f'SELECT DISTINCT {table} . {column} FROM {table}')
                    if column.endswith('number'):
                        self.grammar_dictionary[f'{table}_{column}_string'] = \
                                sorted([f'"{str(row[0])}"' for row in self.cursor.fetchall()], reverse=True)
                    else:
                        self.grammar_dictionary[f'{table}_{column}_string'] = \
                                sorted([f'"\'{str(row[0])}\'"' for row in self.cursor.fetchall()], reverse=True)

        self.grammar_dictionary['biexpr'] = sorted(biexprs, reverse=True) + \
                ['( col_ref ws binaryop ws value)', '(value ws binaryop ws value)']
        return '\n'.join([f"{nonterminal} = {' / '.join(right_hand_side)}"
                          for nonterminal, right_hand_side in self.grammar_dictionary.items()])
