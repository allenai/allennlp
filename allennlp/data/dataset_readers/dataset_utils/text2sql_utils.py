
"""
Utility functions for reading the standardised text2sql datasets presented in
`"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
"""
from typing import List, Dict, NamedTuple, Iterable, Tuple, Set

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
    variable_tags : ``List[str]``
        Labels for each word in ``text`` which correspond to
        which variable in the sql the token is linked to. "O"
        is used to denote no tag.
    sql : ``List[str]``
        The tokens in the SQL query which corresponds to the text.
    text_variables : ``Dict[str, str]``
        A dictionary of variables associated with the text, e.g. {"city_name0": "san fransisco"}
    sql_variables : ``Dict[str, str]``
        A dictionary of variables associated with the sql query.
    """
    text: List[str]
    text_with_variables: List[str]
    variable_tags: List[str]
    sql: List[str]
    text_variables: Dict[str, str]
    sql_variables: Dict[str, str]


def replace_variables(sentence: List[str],
                      sentence_variables: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Replaces abstract variables in text with their concrete counterparts.
    """
    tokens = []
    tags = []
    for token in sentence:
        if token not in sentence_variables:
            tokens.append(token)
            tags.append("O")
        else:
            for word in sentence_variables[token].split():
                tokens.append(word)
                tags.append(token)
    return tokens, tags

def clean_and_split_sql(sql: str) -> List[str]:
    """
    Cleans up and unifies a SQL query. This involves unifying quoted strings
    and splitting brackets which aren't formatted consistently in the data.
    """
    sql_tokens = []
    for token in sql.strip().split():
        token = token.replace('"', "'").replace("%", "")
        if token.endswith("(") and len(token) > 1:
            sql_tokens.append(token[:-1])
            sql_tokens.append(token[-1])
        else:
            sql_tokens.append(token)
    return sql_tokens


def process_sql_data(data: List[JsonDict],
                     use_all_sql: bool = False,
                     use_all_queries: bool = False) -> Iterable[SqlData]:
    """
    A utility function for reading in text2sql data. The blob is
    the result of loading the json from a file produced by the script
    ``scripts/reformat_text2sql_data.py``.

    Parameters
    ----------
    data : ``JsonDict``
    use_all_sql : ``bool``, optional (default = False)
        Whether to use all of the sql queries which have identical semantics,
        or whether to just use the first one.
    use_all_queries : ``bool``, (default = False)
        Whether or not to enforce query sentence uniqueness. If false,
        duplicated queries will occur in the dataset as separate instances,
        as for a given SQL query, not only are there multiple queries with
        the same template, but there are also duplicate queries.
    """
    for example in data:
        seen_sentences: Set[str] = set()
        for sent_info in example['sentences']:
            # Loop over the different sql statements with "equivalent" semantics
            for sql in example["sql"]:
                text_with_variables = sent_info['text'].strip().split()
                text_vars = sent_info['variables']

                query_tokens, tags = replace_variables(text_with_variables, text_vars)
                if not use_all_queries:
                    key = " ".join(query_tokens)
                    if key in seen_sentences:
                        continue
                    else:
                        seen_sentences.add(key)

                sql_tokens = clean_and_split_sql(sql)
                sql_variables = {}
                for variable in example['variables']:
                    sql_variables[variable['name']] = variable['example']

                sql_data = SqlData(text=query_tokens,
                                   text_with_variables=text_with_variables,
                                   variable_tags=tags,
                                   sql=sql_tokens,
                                   text_variables=text_vars,
                                   sql_variables=sql_variables)
                yield sql_data

                # Some questions might have multiple equivalent SQL statements.
                # By default, we just use the first one. TODO(Mark): Use the shortest?
                if not use_all_sql:
                    break
