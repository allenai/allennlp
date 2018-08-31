
from typing import List, Dict
import json

from allennlp.common import JsonDict


def get_template(sql_tokens: List[str],
                 sql_variables: Dict[str, str],
                 sent_variables: Dict[str, str]):
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
    return template

def get_tokens_and_tags(sentence: List[str],
                        sent_variables: Dict[str, str],
                        replace_all_variables: bool = False):
    """
    sentence: The string of the sentence.
    sent_variables: The variable in the sentence and it's actual string. e.g {'var0': 'texas'}
    """
    tokens = []
    tags = []
    for token in sentence:
        if (token not in sent_variables) or replace_all_variables:
            tokens.append(token)
            tags.append("O")
        else:
            for word in sent_variables[token].split():
                tokens.append(word)
                tags.append(token)
    return tokens, tags

def insert_variables(sql: str,
                     sql_variables: List[Dict[str, str]],
                     sent: str,
                     sent_variables: Dict[str, str]):
    """
    sql: The string sql query.
    sql_variables: The variables extracted from the sentence and sql query.
    e.g. [{'example': 'arizona', 'location': 'both', 'name': 'var0', 'type': 'state'}]
    sent: The string of the sentence.
    sent_variables: The variable in the sentence and it's actual string. e.g {'var0': 'texas'}
    """
    split_sentence = sent.strip().split()
    tokens, tags = get_tokens_and_tags(split_sentence, sent_variables)

    sql_tokens = []
    for token in sql.strip().split():
        token = token.replace('"', "'").replace("%", "")
        if token.endswith("(") and len(token) > 1:
            sql_tokens.append(token[:-1])
            sql_tokens.append(token[-1])
        else:
            sql_tokens.append(token)

    template = get_template(sql_tokens, sql_variables, sent_variables)

    return (tokens, tags, ' '.join(template))

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
            # By default, we just use the first one. TODO(MARK) - Use the shortest?
            if not use_all_sql:
                break


def get_train_dev_test_splits(filename: str):
    train = []
    dev = []
    test = []
    with open(filename) as input_file:
        data = json.load(input_file)
        if not isinstance(data, list):
            data = [data]
        for example in data:
            for dataset, instance in get_tagged_data_for_query(example):
                if dataset == 'train':
                    train.append(instance)
                elif dataset == 'dev':
                    dev.append(instance)
                elif dataset == 'test':
                    test.append(instance)
    return train, dev, test
