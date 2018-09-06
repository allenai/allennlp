
"""
Utility functions for reading the standardised text2sql datasets presented in
`"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
"""
from typing import List, Dict, Tuple, NamedTuple, Iterable
import json

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


def get_tokens(sentence: List[str],
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
    Cleans up and unifies a SQL query. This involves removing uncessary quotes
    and spliting brackets which aren't formatted consistently in the data.
    """
    sql_tokens = []
    for token in sql.strip().split():
        token = token.replace('"', "").replace('"', "").replace("%", "")
        if token.endswith("(") and len(token) > 1:
            sql_tokens.append(token[:-1])
            sql_tokens.append(token[-1])
        else:
            sql_tokens.append(token)
    return sql_tokens


def process_sql_data_blob(data: JsonDict,
                          use_all_sql: bool = False,
                          use_question_split: bool = False,
                          cross_validation_split: int = None) -> Iterable[Tuple[str, SqlData]]:
    # If we're splitting based on SQL queries,
    # we assign whole splits of questions which
    # have a similar SQL template to the same split.
    dataset_split: str = data['query-split']

    # TODO(Mark): currently this does not filter for duplicate _sentences_
    # which have the same sql query. Really it should, because these instances
    # are literally identical, so just magnify errors etc. However, doing this
    # would make it really hard to compare to previous work. Sad times.
    for sent_info in data['sentences']:
        # Instead, if we're using the question split,
        # we take the split according to the individual question.
        if use_question_split:
            dataset_split = sent_info['question-split']
        # We are observing the split we're meant to use for cross-validation.
        # set the dataset split to be test. NOTE: This was _incorrect_ in the
        # original repo, causing test leakage in datasets which used cross-validation.
        if cross_validation_split is not None:
            if str(cross_validation_split) == str(dataset_split):
                dataset_bucket = "test"
            else:
                dataset_bucket = "train"
        else:
            dataset_bucket = dataset_split
        # Loop over the different sql statements with "equivelent" semantics
        for sql in data["sql"]:
            sql_variables = {}
            for variable in data['variables']:
                sql_variables[variable['name']] = variable['example']

            text_with_variables = sent_info['text'].strip().split()
            text_vars = sent_info['variables']

            query_tokens = get_tokens(text_with_variables, text_vars)
            sql_tokens = clean_and_split_sql(sql)

            sql_data = SqlData(text=query_tokens,
                               text_with_variables=text_with_variables,
                               sql=sql_tokens,
                               text_variables=text_vars,
                               sql_variables=sql_variables)
            yield (dataset_bucket, sql_data)

            # Some questions might have multiple equivelent SQL statements.
            # By default, we just use the first one. TODO(Mark): Use the shortest?
            if not use_all_sql:
                break

def get_split(filename: str,
              dataset_split: str,
              use_all_sql: bool = False,
              use_question_split: bool = False,
              cross_validation_split: int = None):
    instances = []
    with open(filename) as input_file:
        data = json.load(input_file)
        if not isinstance(data, list):
            data = [data]
        for example in data:
            tagged_example = process_sql_data_blob(example,
                                                   use_all_sql,
                                                   use_question_split,
                                                   cross_validation_split)
            for dataset, instance in tagged_example:
                if dataset == dataset_split:
                    instances.append(instance)
    return instances
