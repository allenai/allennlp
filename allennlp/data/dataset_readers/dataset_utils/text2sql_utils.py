
"""
Utility functions for reading the standardised text2sql datasets presented in
`"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
"""
from typing import List, Dict, Tuple
import json

from allennlp.common import JsonDict


def get_template(sql_tokens: List[str],
                 sql_variables: Dict[str, str],
                 sent_variables: Dict[str, str]):
    """
    Parameters
    ----------
    sentence : ``List[str]``, required.
        The tokens in the sentence.
    sentence_variables : ``Dict[str, str]``, required.
        The variable in the sentence and it's actual string. e.g {'var0': 'texas'}.
    sql_variables : ``Dict[str, str]``, required.
        The variables extracted from the sentence and sql query.
        e.g. {'example': 'arizona', 'location': 'both', 'name': 'var0', 'type': 'state'}

    Returns
    -------
    A string template with keywords replaced with variables coresponding to column names
    in some SQL table.
    """
    template = []
    for token in sql_tokens:
        if (token not in sent_variables) and (token not in sql_variables):
            template.append(token)
        elif token in sent_variables:
            # This is the case that we have the variable
            # in the sql variables but not the sentence variables.
            # Apparently this is denoted with a "".
            if sent_variables[token] == '':
                template.append(sql_variables[token])
            else:
                template.append(token)
        elif token in sql_variables:
            template.append(sql_variables[token])
    return " ".join(template)

def get_tokens_and_tags(sentence: List[str],
                        sentence_variables: Dict[str, str],
                        replace_all_variables: bool = False) -> Tuple[List[str], List[str]]:
    """
    Parameters
    ----------
    sentence : ``List[str]``, required.
        The tokens in the sentence.
    sentence_variables : ``Dict[str, str]``, required.
        The variable in the sentence and it's actual string. e.g {'var0': 'texas'}.
    replace_all_variables : ``bool``, optional (default = False)
        Whether to replace all the variables in the sentence with their corresponding text.

    Returns
    -------
    tokens : ``List[str]``
        The tokens in the sentence with (possibly) variables replaced.
    tags : ``List[str]``
        The tags in the sentence denoting variables.
    """
    tokens = []
    tags = []
    for token in sentence:
        if (token not in sentence_variables) or replace_all_variables:
            tokens.append(token)
            tags.append("O")
        else:
            for word in sentence_variables[token].split():
                tokens.append(word)
                tags.append(token)
    return tokens, tags

def insert_variables(sql: str,
                     sql_variables: List[Dict[str, str]],
                     sentence: str,
                     sentence_variables: Dict[str, str]) -> Tuple[List[str], List[str], str]:
    """
    Parameters
    ----------
    sql : ``str``, required.
        The string sql query.
    sql_variables : ``List[Dict[str, str]]``, required.
        The variables extracted from the sentence and sql query.
        e.g. [{'example': 'arizona', 'location': 'both', 'name': 'var0', 'type': 'state'}]
    sentence : ``str``, required.
        The string of the sentence.
    sentence_variables : ``Dict[str, str]``, required.
        The variable in the sentence and it's actual string. e.g {'var0': 'texas'}.

    Returns
    -------
    The tokens in the sentence, tags denoting variable keywords in the sentence
    and a template to fill.

    """
    split_sentence = sentence.strip().split()
    tokens, tags = get_tokens_and_tags(split_sentence, sentence_variables)

    sql_tokens = []
    for token in sql.strip().split():
        token = token.replace('"', "'").replace("%", "")
        if token.endswith("(") and len(token) > 1:
            sql_tokens.append(token[:-1])
            sql_tokens.append(token[-1])
        else:
            sql_tokens.append(token)

    template = get_template(sql_tokens, sql_variables, sentence_variables)

    return (tokens, tags, template)

def get_tagged_data_for_query(data: JsonDict,
                              use_all_sql: bool = False,
                              use_question_split: bool = False,
                              cross_validation_split: int = None):
    # If we're splitting based on SQL queries,
    # we assign whole splits of questions which
    # have a similar SQL template to the same split.
    dataset_split: str = data['query-split']
    for sent_info in data['sentences']:
        # Instead, if we're using the question split,
        # we take the split according to the individual question.
        if use_question_split:
            dataset_split = sent_info['question-split']

        # We are observing the split we're meant to use for cross-validation.
        # set the dataset split to be test.
        if cross_validation_split is not None:
            if str(cross_validation_split) == str(dataset_split):
                dataset_split = "test"
            else:
                dataset_split = "train"

        # Loop over the different sql statements with "equivelent" semantics
        for sql in data["sql"]:
            sql_variables = {}
            for variable in data['variables']:
                sql_variables[variable['name']] = variable['example']

            text = sent_info['text']
            text_vars = sent_info['variables']

            yield (dataset_split, insert_variables(sql, sql_variables, text, text_vars))

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
            tagged_example = get_tagged_data_for_query(example,
                                                       use_all_sql,
                                                       use_question_split,
                                                       cross_validation_split)
            for dataset, instance in tagged_example:
                if dataset == dataset_split:
                    instances.append(instance)
    return instances
