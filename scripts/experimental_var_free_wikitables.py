import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from allennlp.semparse.contexts.sql_table_context import SqlVisitor
from parsimonious.grammar import Grammar

WIKI_GRAMMAR = Grammar(
        r"""
        statement           = (expr ws statement ws) / (expr)
        expr                = "(" ws expr_contents ws ")"
        expr_contents       = hop_expr /
                              row_list_return_expr /
                              position_expr /
                              count_expr /
                              function_expr /
                              mode_expr /
                              diff_expr

        hop_expr            = ("hop" wsp row_list wsp column)

        row_list_return_expr = paren_row_list_return_expr /
                               arg_expr /
                               filter_numeric_expr /
                               filter_string_expr /
                               same_as_expr
        paren_row_list_return_expr = "(" ws row_list_return_expr ")"

        arg_expr            = (arg_function wsp row_list wsp column)
        arg_function        = "argmin" / "argmax"

        filter_numeric_expr = (numeric_function wsp row_list wsp numeric_value wsp numeric_column)
        numeric_function    = "filter_greater" / "filter_greater_equal" /
                              "filter_less" / "filter_less_equal" /
                              "filter_equal" / "filter_not_equal"

        filter_string_expr  = (string_function wsp row_list wsp string_value wsp string_column)
        string_function     = "filter_in" / "filter_not_in"

        position_expr       = paren_position_expr / (position_function wsp row_list)
        paren_position_expr = "(" ws position_expr ")"
        position_function   = "first" / "last" / "next" / "previous"

        count_expr          = ("count" wsp row_list)

        function_expr       = (function wsp row_list wsp numeric_column)
        function            = "max" / "min" / "average" / "sum"

        mode_expr           = ("mode" wsp row_list wsp column)
        same_as_expr        = ("same_as" wsp row wsp column)
        diff_expr           = ("diff" wsp row wsp row wsp numeric_column)

        column              = numeric_column / string_column
        numeric_column      = (ws "r." name "-num" ws)
        string_column       = (ws "r." name "-str" ws)
        row_list            = row_list_return_expr / ("[" ws internal_row_list ws "]") / "all_rows"
        internal_row_list   = (row wsp internal_row_list ws) / (row)

        row                 = position_expr / name
        numeric_value       = ~"\d*\.?\d+"i
        string_value        = ~"\'.*?\'"i

        name                = ~"[a-zA-Z]\w*"i
        ws                  = ~"\s*"i
        wsp                 = ~"\s+"i
        """
)


def parse_example(example: str):

    sql_visitor = SqlVisitor(WIKI_GRAMMAR)
    try:
        prod_rules = sql_visitor.parse(example)
        print(prod_rules)
    except Exception as e:
        print()
        print(e)
        print(example)

def main() -> None:
    examples = ["(filter_equal [hawaii tacoma] 4 r.score-num)",
                "(mode all_rows r.goals_scored-num)",
                "(count (filter_greater all_rows 46 r.teams-num))",
                "(hop (filter_equal [hawaii tacoma] 4 r.score-num) r.name-str)"]
    for i, example in enumerate(examples):
        print(f"Example {i}")
        parse_example(example)
        print()

if __name__ == "__main__":

    main()
