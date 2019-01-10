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
GRAMMAR_DICTIONARY["query"] = ['(ws select_core ws groupby_clause ws orderby_clause ws limit)',
                               '(ws select_core ws groupby_clause ws orderby_clause)',
                               '(ws select_core ws groupby_clause ws limit)',
                               '(ws select_core ws orderby_clause ws limit)',
                               '(ws select_core ws groupby_clause)',
                               '(ws select_core ws orderby_clause)',
                               '(ws select_core)']

GRAMMAR_DICTIONARY["select_core"] = ['(select_with_distinct ws select_results ws from_clause ws where_clause)',
                                     '(select_with_distinct ws select_results ws from_clause)',
                                     '(select_with_distinct ws select_results ws where_clause)',
                                     '(select_with_distinct ws select_results)']
GRAMMAR_DICTIONARY["select_with_distinct"] = ['(ws "SELECT" ws "DISTINCT")', '(ws "SELECT")']
GRAMMAR_DICTIONARY["select_results"] = ['(ws select_result ws "," ws select_results)', '(ws select_result)']
GRAMMAR_DICTIONARY["select_result"] = ['"*"', '(table_name ws ".*")',
                                       '(expr ws "AS" wsp name)', 'expr', '(col_ref ws "AS" wsp name)']

GRAMMAR_DICTIONARY["from_clause"] = ['ws "FROM" ws source']
GRAMMAR_DICTIONARY["source"] = ['(ws single_source ws "," ws source)', '(ws single_source)']
GRAMMAR_DICTIONARY["single_source"] = ['source_table', 'source_subq']
GRAMMAR_DICTIONARY["source_table"] = ['(table_name ws "AS" wsp name)', 'table_name']
GRAMMAR_DICTIONARY["source_subq"] = ['("(" ws query ws ")" ws "AS" ws name)', '("(" ws query ws ")")']
GRAMMAR_DICTIONARY["limit"] = ['("LIMIT" ws "1")', '("LIMIT" ws number)']

GRAMMAR_DICTIONARY["where_clause"] = ['(ws "WHERE" wsp expr ws where_conj)', '(ws "WHERE" wsp expr)']
GRAMMAR_DICTIONARY["where_conj"] = ['(ws "AND" wsp expr ws where_conj)', '(ws "AND" wsp expr)']

GRAMMAR_DICTIONARY["groupby_clause"] = ['(ws "GROUP" ws "BY" ws group_clause ws "HAVING" ws expr)',
                                        '(ws "GROUP" ws "BY" ws group_clause)']
GRAMMAR_DICTIONARY["group_clause"] = ['(ws expr ws "," ws group_clause)', '(ws expr)']

GRAMMAR_DICTIONARY["orderby_clause"] = ['ws "ORDER" ws "BY" ws order_clause']
GRAMMAR_DICTIONARY["order_clause"] = ['(ordering_term ws "," ws order_clause)', 'ordering_term']
GRAMMAR_DICTIONARY["ordering_term"] = ['(ws expr ws ordering)', '(ws expr)']
GRAMMAR_DICTIONARY["ordering"] = ['(ws "ASC")', '(ws "DESC")']

GRAMMAR_DICTIONARY["col_ref"] = ['(table_name ws "." ws column_name)', 'table_name']
GRAMMAR_DICTIONARY["table_name"] = ['name']
GRAMMAR_DICTIONARY["column_name"] = ['name']
GRAMMAR_DICTIONARY["ws"] = [r'~"\s*"i']
GRAMMAR_DICTIONARY['wsp'] = [r'~"\s+"i']
GRAMMAR_DICTIONARY['name'] = [r'~"[a-zA-Z]\w*"i']

GRAMMAR_DICTIONARY["expr"] = ['in_expr',
                              # Like expressions.
                              '(value wsp "LIKE" wsp string)',
                              # Between expressions.
                              '(value ws "BETWEEN" wsp value ws "AND" wsp value)',
                              # Binary expressions.
                              '(value ws binaryop wsp expr)',
                              # Unary expressions.
                              '(unaryop ws expr)',
                              # Two types of null check expressions.
                              '(col_ref ws "IS" ws "NOT" ws "NULL")',
                              '(col_ref ws "IS" ws "NULL")',
                              'source_subq',
                              'value']
GRAMMAR_DICTIONARY["in_expr"] = ['(value wsp "NOT" wsp "IN" wsp string_set)',
                                 '(value wsp "IN" wsp string_set)',
                                 '(value wsp "NOT" wsp "IN" wsp expr)',
                                 '(value wsp "IN" wsp expr)']

GRAMMAR_DICTIONARY["value"] = ['parenval', '"YEAR(CURDATE())"', 'number', 'boolean',
                               'function', 'col_ref', 'string']
GRAMMAR_DICTIONARY["parenval"] = ['"(" ws expr ws ")"']
GRAMMAR_DICTIONARY["function"] = ['(fname ws "(" ws "DISTINCT" ws arg_list_or_star ws ")")',
                                  '(fname ws "(" ws arg_list_or_star ws ")")']

GRAMMAR_DICTIONARY["arg_list_or_star"] = ['arg_list', '"*"']
GRAMMAR_DICTIONARY["arg_list"] = ['(expr ws "," ws arg_list)', 'expr']
 # TODO(MARK): Massive hack, remove and modify the grammar accordingly
GRAMMAR_DICTIONARY["number"] = [r'~"\d*\.?\d+"i', "'3'", "'4'"]
GRAMMAR_DICTIONARY["string_set"] = ['ws "(" ws string_set_vals ws ")"']
GRAMMAR_DICTIONARY["string_set_vals"] = ['(string ws "," ws string_set_vals)', 'string']
GRAMMAR_DICTIONARY["string"] = ['~"\'.*?\'"i']
GRAMMAR_DICTIONARY["fname"] = ['"COUNT"', '"SUM"', '"MAX"', '"MIN"', '"AVG"', '"ALL"']
GRAMMAR_DICTIONARY["boolean"] = ['"true"', '"false"']

# TODO(MARK): This is not tight enough. AND/OR are strictly boolean value operators.
GRAMMAR_DICTIONARY["binaryop"] = ['"+"', '"-"', '"*"', '"/"', '"="', '"<>"',
                                  '">="', '"<="', '">"', '"<"', '"AND"', '"OR"', '"LIKE"']
GRAMMAR_DICTIONARY["unaryop"] = ['"+"', '"-"', '"not"', '"NOT"']



GLOBAL_DATASET_VALUES: Dict[str, List[str]] = {
        # These are used to check values are present, or numbers of authors.
        "scholar": ["0", "1", "2"],
        # 0 is used for "sea level", 750 is a "major" lake, and 150000 is a "major" city.
        "geography": ["0", "750", "150000"],
        # This defines what an "above average" restaurant is.
        "restaurants": ["2.5"]
}


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


def update_grammar_with_global_values(grammar_dictionary: Dict[str, List[str]], dataset_name: str):

    values = GLOBAL_DATASET_VALUES.get(dataset_name, [])
    values_for_grammar = [f'"{str(value)}"' for value in values]
    grammar_dictionary["value"] = values_for_grammar + grammar_dictionary["value"]


def update_grammar_to_be_variable_free(grammar_dictionary: Dict[str, List[str]]):
    """
    SQL is a predominately variable free language in terms of simple usage, in the
    sense that most queries do not create references to variables which are not
    already static tables in a dataset. However, it is possible to do this via
    derived tables. If we don't require this functionality, we can tighten the
    grammar, because we don't need to support aliased tables.
    """

    # Tables in variable free grammars cannot be aliased, so we
    # remove this functionality from the grammar.
    grammar_dictionary["select_result"] = ['"*"', '(table_name ws ".*")', 'expr']

    # Similarly, collapse the definition of a source table
    # to not contain aliases and modify references to subqueries.
    grammar_dictionary["single_source"] = ['table_name', '("(" ws query ws ")")']
    del grammar_dictionary["source_subq"]
    del grammar_dictionary["source_table"]

    grammar_dictionary["expr"] = ['in_expr',
                                  '(value wsp "LIKE" wsp string)',
                                  '(value ws "BETWEEN" wsp value ws "AND" wsp value)',
                                  '(value ws binaryop wsp expr)',
                                  '(unaryop ws expr)',
                                  '(col_ref ws "IS" ws "NOT" ws "NULL")',
                                  '(col_ref ws "IS" ws "NULL")',
                                  # This used to be source_subq - now
                                  # we don't need aliases, we can colapse it to queries.
                                  '("(" ws query ws ")")',
                                  'value']

    # Finally, remove the ability to reference an arbitrary name,
    # because now we don't have aliased tables, we don't need
    # to recognise new variables.
    del grammar_dictionary["name"]

def update_grammar_with_untyped_entities(grammar_dictionary: Dict[str, List[str]]) -> None:
    """
    Variables can be treated as numbers or strings if their type can be inferred -
    however, that can be difficult, so instead, we can just treat them all as values
    and be a bit looser on the typing we allow in our grammar. Here we just remove
    all references to number and string from the grammar, replacing them with value.
    """
    grammar_dictionary["string_set_vals"] = ['(value ws "," ws string_set_vals)', 'value']
    grammar_dictionary["value"].remove('string')
    grammar_dictionary["value"].remove('number')
    grammar_dictionary["limit"] = ['("LIMIT" ws "1")', '("LIMIT" ws value)']
    grammar_dictionary["expr"][1] = '(value wsp "LIKE" wsp value)'
    del grammar_dictionary["string"]
    del grammar_dictionary["number"]


def update_grammar_values_with_variables(grammar_dictionary: Dict[str, List[str]],
                                         prelinked_entities: Dict[str, Dict[str, str]]) -> None:

    for variable, _ in prelinked_entities.items():
        grammar_dictionary["value"] = [f'"\'{variable}\'"'] + grammar_dictionary["value"]


def update_grammar_numbers_and_strings_with_variables(grammar_dictionary: Dict[str, List[str]], # pylint: disable=invalid-name
                                                      prelinked_entities: Dict[str, Dict[str, str]],
                                                      columns: Dict[str, TableColumn]) -> None:
    for variable, info in prelinked_entities.items():
        variable_column = info["type"].upper()
        matched_column = columns.get(variable_column, None)

        if matched_column is not None:
            # Try to infer the variable's type by matching it to a column in
            # the database. If we can't, we just add it as a value.
            if column_has_numeric_type(matched_column):
                grammar_dictionary["number"] = [f'"\'{variable}\'"'] + grammar_dictionary["number"]
            elif column_has_string_type(matched_column):
                grammar_dictionary["string"] = [f'"\'{variable}\'"'] + grammar_dictionary["string"]
            else:
                grammar_dictionary["value"] = [f'"\'{variable}\'"'] + grammar_dictionary["value"]
        # Otherwise, try to infer by looking at the actual value:
        else:
            try:
                # This is what happens if you try and do type inference
                # in a grammar which parses _strings_ in _Python_.
                # We're just seeing if the python interpreter can convert
                # to to a float - if it can, we assume it's a number.
                float(info["text"])
                is_numeric = True
            except ValueError:
                is_numeric = False
            if is_numeric:
                grammar_dictionary["number"] = [f'"\'{variable}\'"'] + grammar_dictionary["number"]
            elif info["text"].replace(" ", "").isalpha():
                grammar_dictionary["string"] = [f'"\'{variable}\'"'] + grammar_dictionary["string"]
            else:
                grammar_dictionary["value"] = [f'"\'{variable}\'"'] + grammar_dictionary["value"]
