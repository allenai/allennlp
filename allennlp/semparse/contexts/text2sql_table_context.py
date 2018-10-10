# pylint: disable=anomalous-backslash-in-string
"""
A ``Text2SqlTableContext`` represents the SQL context in which an utterance appears
for the any of the text2sql datasets, with the grammar and the valid actions.
"""
from typing import List, Dict
from sqlite3 import Cursor


from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import TableColumn
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import column_has_numeric_type
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import column_has_string_type

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
GRAMMAR_DICTIONARY["select_with_distinct"] = ['(ws "SELECT" ws "DISTINCT")', '(ws "SELECT")']
GRAMMAR_DICTIONARY["select_results"] = ['(ws select_result ws "," ws select_results)', '(ws select_result)']
GRAMMAR_DICTIONARY["select_result"] = ['sel_res_all_star', 'sel_res_tab_star', 'sel_res_val', 'sel_res_col']
GRAMMAR_DICTIONARY["sel_res_tab_star"] = ['name ".*"']
GRAMMAR_DICTIONARY["sel_res_all_star"] = ['"*"']
GRAMMAR_DICTIONARY['sel_res_val'] = ['(expr ws "AS" wsp name)', 'expr']
GRAMMAR_DICTIONARY['sel_res_col'] = ['col_ref ws "AS" wsp name']

GRAMMAR_DICTIONARY["from_clause"] = ['ws "FROM" ws source']
GRAMMAR_DICTIONARY["source"] = ['(ws single_source ws "," ws source)', '(ws single_source)']
GRAMMAR_DICTIONARY["single_source"] = ['source_table', 'source_subq']
GRAMMAR_DICTIONARY["source_table"] = ['table_name', '(table_name ws "AS" wsp name)']
GRAMMAR_DICTIONARY["source_subq"] = ['("(" ws query ws ")" ws "AS" ws name)', '("(" ws query ws ")")']

GRAMMAR_DICTIONARY["where_clause"] = ['(ws "WHERE" wsp expr where_conj)', '(ws "WHERE" wsp expr)']
GRAMMAR_DICTIONARY["where_conj"] = ['(ws "AND" wsp expr where_conj)', '(ws "AND" wsp expr)']

GRAMMAR_DICTIONARY["groupby_clause"] = ['(ws "GROUP" ws "BY" group_clause ws "HAVING" ws expr)',
                                        '(ws "GROUP" ws "BY" group_clause)']
GRAMMAR_DICTIONARY["group_clause"] = ['(ws expr ws "," group_clause)', '(ws expr)']

GRAMMAR_DICTIONARY["orderby_clause"] = ['ws "ORDER" ws "BY" order_clause']
GRAMMAR_DICTIONARY["order_clause"] = ['(ordering_term ws "," order_clause)', 'ordering_term']
GRAMMAR_DICTIONARY["ordering_term"] = ['(ws expr ordering)', '(ws expr)']
GRAMMAR_DICTIONARY["ordering"] = ['(ws "ASC")', '(ws "DESC")']
GRAMMAR_DICTIONARY["limit"] = ['ws "LIMIT" ws number']

GRAMMAR_DICTIONARY["col_ref"] = ['(table_name ws "." ws column_name)', 'table_name']
GRAMMAR_DICTIONARY["table_name"] = ['name']
GRAMMAR_DICTIONARY["column_name"] = ['name']
GRAMMAR_DICTIONARY["ws"] = ['~"\s*"i']
GRAMMAR_DICTIONARY['wsp'] = ['~"\s+"i']
GRAMMAR_DICTIONARY['name'] = ['~"[a-zA-Z]\w*"i']

GRAMMAR_DICTIONARY["expr"] = ['in_expr', 'like_expr', 'between_expr', 'binary_expr',
                              'unary_expr', 'null_check_expr', 'source_subq', 'value']
GRAMMAR_DICTIONARY["like_expr"] = ['value wsp "LIKE" wsp string']
GRAMMAR_DICTIONARY["in_expr"] = ['(value wsp "NOT" wsp "IN" wsp string_set)',
                                 '(value wsp "IN" wsp string_set)',
                                 '(value wsp "NOT" wsp "IN" wsp expr)',
                                 '(value wsp "IN" wsp expr)']

GRAMMAR_DICTIONARY["between_expr"] = ['value ws "BETWEEN" wsp value ws "AND" wsp value']
GRAMMAR_DICTIONARY["binary_expr"] = ['value ws binaryop wsp expr']
GRAMMAR_DICTIONARY["unary_expr"] = ['unaryop expr']
GRAMMAR_DICTIONARY["null_check_expr"] = ['(col_ref ws "IS" ws "NOT" ws "NULL")', '(col_ref ws "IS" ws "NULL")']

GRAMMAR_DICTIONARY["value"] = ['parenval', '"YEAR(CURDATE())"', 'number', 'boolean',
                               'function', 'col_ref', 'string']
GRAMMAR_DICTIONARY["parenval"] = ['"(" ws expr ws ")"']
GRAMMAR_DICTIONARY["function"] = ['(fname ws "(" ws "DISTINCT" ws arg_list_or_star ws ")")',
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
                                  '">="', '"<="', '">"', '"<"', '"AND"', '"OR"', '"LIKE"']
GRAMMAR_DICTIONARY["unaryop"] = ['"+"', '"-"', '"not"', '"NOT"']

def update_grammar_with_tables(grammar_dictionary: Dict[str, List[str]],
                               schema: Dict[str, List[TableColumn]]) -> None:
    table_names = sorted([f'"{table}"' for table in
                          list(schema.keys())], reverse=True)
    grammar_dictionary['table_name'] = table_names

    all_columns = set()
    for table in schema.values():
        all_columns.update([column.name for column in table])
    sorted_columns = sorted([f'"{column}"' for column in all_columns], reverse=True)
    grammar_dictionary['column_name'] = sorted_columns

def update_grammar_with_table_values(grammar_dictionary: Dict[str, List[str]],
                                     schema: Dict[str, List[TableColumn]],
                                     cursor: Cursor) -> None:

    for table_name, columns in schema.items():
        for column in columns:
            cursor.execute(f'SELECT DISTINCT {table_name}.{column.name} FROM {table_name}')
            results = [x[0] for x in cursor.fetchall()]
            if column_has_string_type(column):
                productions = sorted([f'"{str(result)}"' for result in results], reverse=True)
                grammar_dictionary["string"].extend(productions)
            elif column_has_numeric_type(column):
                productions = sorted([f'"{str(result)}"' for result in results], reverse=True)
                grammar_dictionary["number"].extend(productions)
