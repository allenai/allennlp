import json
import os
import sys
from typing import Any, Iterable, Tuple
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import process_sql_data, SqlData
from allennlp.semparse.contexts.sql_table_context import SqlVisitor
from parsimonious.grammar import Grammar

# Changes:
# Added AVG, SUM to agg_func
# added optional AS to table_name
# Made where_clause optional in query
# Add query to pos_value (TODO check this, very unclear if it is the correct way to handle this)
# Added optional DISTINCT inside agg
# added <> binary op
# added biexpr to agg to support "SELECT TABLE.COLUMN / TABLE2.COLUMN2 FROM ..."
# added optional extra agg clauses connected to each other by a binaryop, fixes e.g SUM ( STATEalias0.POPULATION ) / SUM ( STATEalias0.AREA )
# Added optional nested brackets inside DISTINCT for aggregates (common in yelp)

# still TODO: 
# GROUP BY, ORDER BY
# think about the  difference between variable and string matching
# tablename in (tablename1, tablename2)
# JOIN, seems hard.

# TODO:
# string set isn't an expr, it should only be in in_expr
# not all functions can take * as an argument.    

SQL_GRAMMAR2 = Grammar(
        r"""
        stmt             = (query ws ";") / (query ws)
        query            = (ws select_core groupby_clause orderby_clause ws limit) /
                           (ws select_core groupby_clause ws limit) /
                           (ws select_core orderby_clause ws limit) /
                           (ws select_core groupby_clause) /
                           (ws select_core orderby_clause) /
                           (ws select_core)

        select_core      = (select_with_distinct select_results from_clause where_clause) /
                           (select_with_distinct select_results from_clause) /
                           (select_with_distinct select_results where_clause) /
                           (select_with_distinct select_results)

        select_with_distinct = (SELECT DISTINCT) / SELECT
        select_results   = (ws select_result ws "," ws select_results) / (ws select_result)
        select_result    = sel_res_all_star / sel_res_tab_star / sel_res_val / sel_res_col

        sel_res_tab_star = name ".*"
        sel_res_all_star = "*"
        sel_res_val      = (expr AS wsp name) / expr
        sel_res_col      = col_ref (AS wsp name)

        from_clause      = FROM source
        source           = (ws single_source ws "," ws source) / (ws single_source)
        single_source    = source_table / source_subq
        source_table     = table_name (AS wsp name)
        source_subq      = ("(" ws query ws ")" AS ws name) / ("(" ws query ws ")")
        where_clause     = (WHERE wsp expr where_conj) / (WHERE wsp expr)
        where_conj       = (AND wsp expr where_conj) / (AND wsp expr)

        groupby_clause   = (GROUP BY group_clause having_clause) / (GROUP BY group_clause)
        group_clause     = (ws expr ws "," group_clause) / (ws expr)
        having_clause    = HAVING ws expr

        orderby_clause   = ORDER BY order_clause
        order_clause     = (ordering_term ws "," order_clause) / ordering_term
        ordering_term    = (ws expr ordering) / (ws expr)
        ordering         = ASC / DESC
        limit            = LIMIT ws number

        col_ref          = (table_name "." column_name) / column_name
        table_name       = name
        column_name      = name
        ws               = ~"\s*"i
        wsp              = ~"\s+"i
        name             = ~"[a-zA-Z]\w*"i

        expr             = in_expr / like_expr / between_expr / binary_expr / unary_expr / source_subq / value / string_set
        like_expr        = value wsp LIKE ws string
        in_expr          = (value wsp NOT IN wsp expr) / (value wsp IN wsp expr)
        between_expr     = value BETWEEN wsp value AND wsp value
        binary_expr      = value ws binaryop ws expr
        unary_expr       = unaryop expr
        value            = parenval / number / boolean / function / col_ref / string
        parenval         = "(" ws expr ws ")"
        function         = (fname ws "(" ws DISTINCT ws arg_list_or_star ws ")") /
                           (fname ws "(" ws arg_list_or_star ws ")")

        arg_list_or_star = arg_list / "*"
        arg_list         = (expr ws "," ws arg_list) / expr
        number           = ~"\d*\.?\d+"i
        string_set       = ws "(" ws string_set_vals ws ")"
        string_set_vals  = (string ws "," ws string_set_vals) / string
        string           = ~"\'.*?\'"i
        fname            = "COUNT" / "SUM" / "MAX" / "MIN" / "AVG"
        boolean          = "true" / "false"
        binaryop         = "+" / "-" / "*" / "/" / "=" / "<>" / ">=" / "<=" / ">" / "<" / ">" / "and" / "AND" / "or" / "OR"
        binaryop_no_andor = "+" / "-" / "*" / "/" / "=" / "<>" / "<=" / ">" / "<" / ">"
        unaryop          = "+" / "-" / "not" / "NOT"

        SELECT   = ws "SELECT"
        FROM     = ws "FROM"
        WHERE    = ws "WHERE"
        AS       = ws "AS"
        AND      = ws "AND"
        DISTINCT = ws "DISTINCT"
        GROUP    = ws "GROUP"
        ORDER    = ws "ORDER"
        BY       = ws "BY"
        ASC      = ws "ASC"
        DESC     = ws "DESC"
        BETWEEN  = ws "BETWEEN"
        IN       = ws "IN"
        NOT      = ws "NOT"
        HAVING   = ws "HAVING"
        LIMIT    = ws "LIMIT"
        LIKE     = ws "LIKE"
        """
)

def parse_dataset(filename: str, filter_by: str = None, verbose: bool = False):

    filter_by = filter_by or "13754332dvmklfdsaf-3543543"
    data = json.load(open(filename))
    num_queries = 0
    num_parsed = 0
    filtered_errors = 0

    sql_visitor = SqlVisitor(SQL_GRAMMAR2)
    for i, sql_data in enumerate(process_sql_data(data)):
        sql_string = " ".join(sql_data.sql)
        num_queries += 1
        try:
            prod_rules = sql_visitor.parse(sql_string)
            num_parsed += 1
        except Exception as e:

            if filter_by in sql_string:
                filtered_errors += 1

            if verbose and filter_by not in sql_string:
                print()
                print(e)
                print(" ".join(sql_data.text))
                try:
                    import sqlparse
                    print(sqlparse.format(sql_string, reindent=True))
                except Exception:
                    print(sql_string)

        if (i + 1) % 500 == 0:
            print(f"\tProcessed {i + 1} queries.")

    return num_parsed, num_queries, filtered_errors

def main(data_directory: int, dataset: str = None, filter_by: str = None, verbose: bool = False) -> None:
    """
    Parameters
    ----------
    data_directory : str, required.
        The path to the data directory of https://github.com/jkkummerfeld/text2sql-data
        which has been preprocessed using scripts/reformat_text2sql_data.py.
    dataset : str, optional.
        The dataset to parse. By default all are parsed.
    filter_by : str, optional
        Compute statistics about a particular error and only print errors which don't contain this string.
    verbose : bool, optional.
        Whether to print information about incorrectly parsed SQL.
    """
    directory_dict = {path: files for path, names, files in os.walk(data_directory) if files}

    for directory, data_files in directory_dict.items():
        if "query_split" in directory or  (dataset is not None and dataset not in directory):
            continue

        print(f"Parsing dataset at {directory}")
        parsed = 0
        total = 0
        for json_file in data_files:
            print(f"\tParsing split at {json_file}")
            file_path = os.path.join(directory, json_file)
            num_parsed, num_queries, filtered_errors = parse_dataset(file_path, filter_by, verbose)

            parsed += num_parsed
            total += num_queries

        print(f"\tParsed {parsed} out of {total} queries, coverage {parsed/total}")
        if filter_by is not None:
            print(f"\tOf {total - parsed} errors, {filtered_errors/ (total - parsed + 1e-13)} contain {filter_by}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check the coverage of a SQL Grammar on the text2sql datasets.")
    parser.add_argument('--data', type=str, help='The path to the text2sql data directory.')
    parser.add_argument('--dataset', type=str, default=None,
                        help='The dataset to check coverage for. Defaults to all datasets.')
    parser.add_argument('--filter', type=str, default=None, help='A string to filter by.')
    parser.add_argument('--verbose', help='Verbose output.', action='store_true')
    args = parser.parse_args()
    main(args.data, args.dataset, args.filter, args.verbose)
