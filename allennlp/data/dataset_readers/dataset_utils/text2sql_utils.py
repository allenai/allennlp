
"""
Utility functions for reading the standardised text2sql datasets presented in
`"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
"""
from typing import List, Dict, NamedTuple, Iterable

from allennlp.common import JsonDict


class SqlData(NamedTuple):
    """
    A utility class for reading in text2sql data.

    Parameters
    ----------
    text : ``List[str]``
        The tokens in the text of the query.
    text_with_variables : ``List[str]``
        The tokens in the text of the query with variables
        mapped to table names/abstract variables.
    sql : ``List[str]``
        The tokens in the SQL query which corresponds to the text.
    text_variables : ``Dict[str, str]``
        A dictionary of variables associated with the text, e.g. {"city_name0": "san fransisco"}
    sql_variables : ``Dict[str, str]``
        A dictionary of variables associated with the sql query.
    """
    text: List[str]
    text_with_variables: List[str]
    sql: List[str]
    text_variables: Dict[str, str]
    sql_variables: Dict[str, str]


def replace_variables(sentence: List[str],
                      sentence_variables: Dict[str, str]) -> List[str]:
    """
    Replaces abstract variables in text with their concrete counterparts.
    """
    tokens = []
    for token in sentence:
        if token not in sentence_variables:
            tokens.append(token)
        else:
            for word in sentence_variables[token].split():
                tokens.append(word)
    return tokens

def clean_and_split_sql(sql: str) -> List[str]:
    """
    Cleans up and unifies a SQL query. This involves removing unnecessary quotes
    and splitting brackets which aren't formatted consistently in the data.
    """
    sql_tokens = []
    for token in sql.strip().split():
        token = token.replace('"', "").replace("'", "").replace("%", "")
        if token.endswith("(") and len(token) > 1:
            sql_tokens.append(token[:-1])
            sql_tokens.append(token[-1])
        else:
            sql_tokens.append(token)
    return sql_tokens


def process_sql_data_blob(data: JsonDict,
                          use_all_sql: bool = False) -> Iterable[SqlData]:
    """
    A utility function for reading in text2sql data blobs. The blob is
    the result of loading the json from a file produced by the script
    ``scripts/reformat_text2sql_data.py``.

    Parameters
    ----------
    data : ``JsonDict``
    use_all_sql : ``bool``, optional (default = False)
        Whether to use all of the sql queries which have identical semantics,
        or whether to just use the first one.
    """
    # TODO(Mark): currently this does not filter for duplicate _sentences_
    # which have the same sql query. Really it should, because these instances
    # are literally identical, so just magnify errors etc. However, doing this
    # would make it really hard to compare to previous work. Sad times.
    for sent_info in data['sentences']:
        # Loop over the different sql statements with "equivalent" semantics
        for sql in data["sql"]:
            sql_variables = {}
            for variable in data['variables']:
                sql_variables[variable['name']] = variable['example']

            text_with_variables = sent_info['text'].strip().split()
            text_vars = sent_info['variables']

            query_tokens = replace_variables(text_with_variables, text_vars)
            sql_tokens = clean_and_split_sql(sql)

            sql_data = SqlData(text=query_tokens,
                               text_with_variables=text_with_variables,
                               sql=sql_tokens,
                               text_variables=text_vars,
                               sql_variables=sql_variables)
            yield sql_data

            # Some questions might have multiple equivalent SQL statements.
            # By default, we just use the first one. TODO(Mark): Use the shortest?
            if not use_all_sql:
                break
