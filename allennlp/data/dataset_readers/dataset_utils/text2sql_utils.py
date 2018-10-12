
"""
Utility functions for reading the standardised text2sql datasets presented in
`"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
"""
from typing import List, Dict, NamedTuple, Iterable, Tuple, Set
from collections import defaultdict

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
    sql_variables : ``Dict[str, Dict[str, str]]``
        A dictionary of variables and column references associated with the sql query.
    """
    text: List[str]
    text_with_variables: List[str]
    variable_tags: List[str]
    sql: List[str]
    text_variables: Dict[str, str]
    sql_variables: Dict[str, Dict[str, str]]

class TableColumn(NamedTuple):
    name: str
    column_type: str
    is_primary_key: bool

def column_has_string_type(column: TableColumn) -> bool:
    if "varchar" in column.column_type:
        return True
    elif column.column_type == "text":
        return True
    elif column.column_type == "longtext":
        return True

    return False

def column_has_numeric_type(column: TableColumn) -> bool:
    if "int" in column.column_type:
        return True
    elif "float" in column.column_type:
        return True
    elif "double" in column.column_type:
        return True
    return False

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

def split_table_and_column_names(table: str) -> Iterable[str]:
    partitioned = [x for x in table.partition(".") if x != '']
    # Avoid splitting decimal strings.
    if partitioned[0].isnumeric() and partitioned[-1].isnumeric():
        return [table]
    return partitioned

def clean_and_split_sql(sql: str) -> List[str]:
    """
    Cleans up and unifies a SQL query. This involves unifying quoted strings
    and splitting brackets which aren't formatted consistently in the data.
    """
    sql_tokens: List[str] = []
    for token in sql.strip().split():
        token = token.replace('"', "'").replace("%", "")
        if token.endswith("(") and len(token) > 1:
            sql_tokens.extend(split_table_and_column_names(token[:-1]))
            sql_tokens.extend(split_table_and_column_names(token[-1]))
        else:
            sql_tokens.extend(split_table_and_column_names(token))
    return sql_tokens

def resolve_primary_keys_in_schema(sql_tokens: List[str],
                                   schema: Dict[str, List[TableColumn]]) -> List[str]:
    """
    Some examples in the text2sql datasets use ID as a column reference to the
    column of a table which has a primary key. This causes problems if you are trying
    to constrain a grammar to only produce the column names directly, because you don't
    know what ID refers to. So instead of dealing with that, we just replace it.
    """
    primary_keys_for_tables = {name: max(columns, key=lambda x: x.is_primary_key).name
                               for name, columns in schema.items()}
    resolved_tokens = []
    for i, token in enumerate(sql_tokens):
        if i > 2:
            table_name = sql_tokens[i - 2]
            if token == "ID" and table_name in primary_keys_for_tables.keys():
                token = primary_keys_for_tables[table_name]
        resolved_tokens.append(token)
    return resolved_tokens

def clean_unneeded_aliases(sql_tokens: List[str]) -> List[str]:

    unneeded_aliases = {}
    previous_token = sql_tokens[0]
    for (token, next_token) in zip(sql_tokens[1:-1], sql_tokens[2:]):
        if token == "AS" and previous_token is not None:
            # Check to see if the table name without the alias
            # is the same.
            table_name = next_token[:-6]
            if table_name == previous_token:
                # If so, store the mapping as a replacement.
                unneeded_aliases[next_token] = previous_token

        previous_token = token

    dealiased_tokens: List[str] = []
    for token in sql_tokens:
        new_token = unneeded_aliases.get(token, None)

        if new_token is not None and dealiased_tokens[-1] == "AS":
            dealiased_tokens.pop()
            continue
        elif new_token is None:
            new_token = token

        dealiased_tokens.append(new_token)

    return dealiased_tokens

def read_dataset_schema(schema_path: str) -> Dict[str, List[TableColumn]]:
    """
    Reads a schema from the text2sql data, returning a dictionary
    mapping table names to their columns and respective types.
    This handles columns in an arbitrary order and also allows
    either ``{Table, Field}`` or ``{Table, Field} Name`` as headers,
    because both appear in the data. It also uppercases table and
    column names if they are not already uppercase.

    Parameters
    ----------
    schema_path : ``str``, required.
        The path to the csv schema.

    Returns
    -------
    A dictionary mapping table names to typed columns.
    """
    schema: Dict[str, List[TableColumn]] = defaultdict(list)
    for i, line in enumerate(open(schema_path, "r")):
        if i == 0:
            header = [x.strip() for x in line.split(",")]
        elif line[0] == "-":
            continue
        else:
            data = {key: value for key, value in zip(header, [x.strip() for x in line.split(",")])}

            table = data.get("Table Name", None) or data.get("Table")
            column = data.get("Field Name", None) or data.get("Field")
            is_primary_key = data.get("Primary Key") == "y"
            schema[table.upper()].append(TableColumn(column.upper(), data["Type"], is_primary_key))

    return {**schema}


def process_sql_data(data: List[JsonDict],
                     use_all_sql: bool = False,
                     use_all_queries: bool = False,
                     remove_unneeded_aliases: bool = False,
                     schema: Dict[str, List[TableColumn]] = None) -> Iterable[SqlData]:
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
    remove_unneeded_aliases : ``bool``, (default = False)
        The text2sql data by default creates alias names for `all` tables,
        regardless of whether the table is derived or if it is identical to
        the original (e.g SELECT TABLEalias0.COLUMN FROM TABLE AS TABLEalias0).
        This is not necessary and makes the action sequence and grammar manipulation
        much harder in a grammar based decoder. Note that this does not
        remove aliases which are legitimately required, such as when a new
        table is formed by performing operations on the original table.
    schema : ``Dict[str, List[TableColumn]]``, optional, (default = None)
        A schema to resolve primary keys against. Converts 'ID' column names
        to their actual name with respect to the Primary Key for the table
        in the schema.
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
                if remove_unneeded_aliases:
                    sql_tokens = clean_unneeded_aliases(sql_tokens)
                if schema is not None:
                    sql_tokens = resolve_primary_keys_in_schema(sql_tokens, schema)

                sql_variables = {}
                for variable in example['variables']:
                    sql_variables[variable['name']] = {'text': variable['example'], 'type': variable['type']}

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
