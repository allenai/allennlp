import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import process_sql_data, SqlData
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor, format_grammar_string
from allennlp.semparse.contexts.text2sql_table_context import GRAMMAR_DICTIONARY
from parsimonious.grammar import Grammar
# still TODO: 
# JOIN, seems hard.
# Added query to pos_value - check this, very unclear if it is the correct way to handle this.
# not all functions can take * as an argument.
# Check whether LIKE can take non string arguments (example in scholar dataset)


def parse_dataset(filename: str, filter_by: str = None, verbose: bool = False):

    grammar_string = format_grammar_string(GRAMMAR_DICTIONARY)
    grammar = Grammar(grammar_string)

    filter_by = filter_by or "13754332dvmklfdsaf-3543543"
    data = json.load(open(filename))
    num_queries = 0
    num_parsed = 0
    filtered_errors = 0

    non_basic_as_aliases = 0
    as_count = 0
    queries_with_weird_as = 0

    for i, sql_data in enumerate(process_sql_data(data)):
        sql_visitor = SqlVisitor(grammar)

        if any([x[:7] == "DERIVED"] for x in sql_data.sql):
            # NOTE: DATA hack alert - the geography dataset doesn't alias derived tables consistently,
            # so we fix the data a bit here instead of completely re-working the grammar.
            sql_to_use = []
            for j, token in enumerate(sql_data.sql):
                if token[:7] == "DERIVED" and sql_data.sql[j-1] == ")":
                    sql_to_use.append("AS")
                sql_to_use.append(token)

            previous_token = None
            query_has_weird_as = False
            for j, token in enumerate(sql_to_use[:-1]):

                if token == "AS" and previous_token is not None:

                    table_name = sql_to_use[j + 1][:-6]
                    if table_name != previous_token:
                        non_basic_as_aliases += 1
                        query_has_weird_as = True
                    as_count += 1
                previous_token = token

            if query_has_weird_as:
                queries_with_weird_as += 1


            sql_string = " ".join(sql_to_use)
        else:
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
                print(sql_data.sql)
                try:
                    import sqlparse
                    print(sqlparse.format(sql_string, reindent=True))
                except Exception:
                    print(sql_string)

        if (i + 1) % 500 == 0:
            print(f"\tProcessed {i + 1} queries.")

    return num_parsed, num_queries, filtered_errors, non_basic_as_aliases, as_count, queries_with_weird_as

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
        total_non_aliases = 0
        total_as_count = 0
        total_queries_with_weird_as = 0
        total = 0
        for json_file in data_files:
            print(f"\tParsing split at {json_file}")
            file_path = os.path.join(directory, json_file)
            num_parsed, num_queries, filtered_errors, non_basic_as_aliases, as_count, queries_with_weird_as = parse_dataset(file_path, filter_by, verbose)

            parsed += num_parsed
            total += num_queries
            total_non_aliases += non_basic_as_aliases
            total_as_count += as_count
            total_queries_with_weird_as += queries_with_weird_as

        print(f"\tParsed {parsed} out of {total} queries, coverage {parsed/total}")
        print(f"\tFound {total_non_aliases} out of {total_as_count} non simple AS aliases. percentage: {total_non_aliases/total_as_count}")
        print(f"\tFound {total_queries_with_weird_as} out of {total} queries with > 1 weird AS. percentage: {total_queries_with_weird_as/total}")
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
