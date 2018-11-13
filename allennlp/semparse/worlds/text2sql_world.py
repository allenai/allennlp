from typing import List, Tuple, Dict
from copy import deepcopy
from collections import defaultdict
import sqlite3
from sqlite3 import Cursor
import os

from parsimonious import Grammar
from parsimonious.exceptions import ParseError
from nltk import ngrams, bigrams

from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import TableColumn
from allennlp.semparse.contexts.sql_context_utils import format_grammar_string, initialize_valid_actions
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import read_dataset_schema
from allennlp.semparse.contexts.text2sql_table_context import GRAMMAR_DICTIONARY
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_table_values
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_tables
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_global_values
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_to_be_variable_free
from allennlp.semparse.contexts.text2sql_table_context import remove_number_and_string_types
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_values_with_variables
from allennlp.data.tokenizers import Token

class Text2SqlWorld(Registrable):

    def get_action_sequence_and_all_actions(self,
                                            query: List[str] = None,
                                            prelinked_entities: Dict[str, Dict[str, str]] = None) -> Tuple[List[str], List[str]]: # pylint: disable=line-too-long
        raise NotImplementedError

    def is_global_rule(self, production_rule: str) -> bool:

        raise NotImplementedError


def database_to_dict(schema: Dict[str, List[TableColumn]],
                     cursor: Cursor) -> Dict[Tuple[str, str], List[str]]:
    database_dict = defaultdict(list)
    for table_name, columns in schema.items():
        for column in columns:
            cursor.execute(f'SELECT DISTINCT {table_name}.{column.name} FROM {table_name}')
            results = [str(x[0]) for x in cursor.fetchall()]
            database_dict[(table_name, column)] = results
    return {**database_dict}

# def get_strings_from_utterance(tokenized_utterance: List[Token]) -> Dict[str, List[int]]:
#     """
#     Based on the current utterance, return a dictionary where the keys are the strings in
#     the database that map to lists of the token indices that they are linked to.
#     """
#     string_linking_scores: Dict[str, List[int]] = defaultdict(list)

#     for index, token in enumerate(tokenized_utterance):
#         for string in ATIS_TRIGGER_DICT.get(token.text.lower(), []):
#             string_linking_scores[string].append(index)

#     token_bigrams = bigrams([token.text for token in tokenized_utterance])
#     for index, token_bigram in enumerate(token_bigrams):
#         for string in ATIS_TRIGGER_DICT.get(' '.join(token_bigram).lower(), []):
#             string_linking_scores[string].extend([index,
#                                                   index + 1])

#     trigrams = ngrams([token.text for token in tokenized_utterance], 3)
#     for index, trigram in enumerate(trigrams):
#         if trigram[0] == 'st':
#             natural_language_key = f'st. {trigram[2]}'.lower()
#         else:
#             natural_language_key = ' '.join(trigram).lower()
#         for string in ATIS_TRIGGER_DICT.get(natural_language_key, []):
#             string_linking_scores[string].extend([index,
#                                                   index + 1,
#                                                   index + 2])
#     return string_linking_scores


@Text2SqlWorld.register("prelinked")
class PrelinkedText2SqlWorld(Text2SqlWorld):
    """
    A World representation for any of the Text2Sql datasets which assumes
    access to pre-linked entities. It does not leverage a database to
    link entities in the question.

    Parameters
    ----------
    use_untyped_entities : ``bool``, optional (default = False)
        Whether or not to try to infer the types of prelinked variables.
        If not, they are added as untyped values to the grammar instead.
    """
    def __init__(self,
                 schema_path: str,
                 use_untyped_entities: bool = False) -> None:
        self.schema = read_dataset_schema(schema_path)
        self.columns = {column.name: column for table in self.schema.values() for column in table}
        self.dataset_name = os.path.basename(schema_path).split("-")[0]
        self.use_untyped_entities = use_untyped_entities

        # NOTE: This base dictionary should not be modified.
        self.base_grammar_dictionary = self._initialize_grammar_dictionary(deepcopy(GRAMMAR_DICTIONARY))

    def get_action_sequence_and_all_actions(self,
                                            query: List[str] = None,
                                            prelinked_entities: Dict[str, Dict[str, str]] = None) -> Tuple[List[str], List[str]]: # pylint: disable=line-too-long
        grammar_with_context = deepcopy(self.base_grammar_dictionary)


        prelinked_entities = prelinked_entities or {}
        if self.use_untyped_entities:
            update_grammar_values_with_variables(grammar_with_context, prelinked_entities)
        else:
            pass
            # TODO here we should update based on _column productions_.

        grammar = Grammar(format_grammar_string(grammar_with_context))

        valid_actions = initialize_valid_actions(grammar)
        all_actions = set()
        for action_list in valid_actions.values():
            all_actions.update(action_list)
        sorted_actions = sorted(all_actions)

        sql_visitor = SqlVisitor(grammar)
        try:
            action_sequence = sql_visitor.parse(" ".join(query)) if query else []
        except ParseError:
            action_sequence = None

        return action_sequence, sorted_actions, None

    def _initialize_grammar_dictionary(self, grammar_dictionary: Dict[str, List[str]]) -> Dict[str, List[str]]:
        # Add all the table and column names to the grammar.
        update_grammar_with_tables(grammar_dictionary, self.schema)

        # Finally, update the grammar with global, non-variable values
        # found in the dataset, if present.
        update_grammar_with_global_values(grammar_dictionary, self.dataset_name)

        update_grammar_to_be_variable_free(grammar_dictionary)

        if self.use_untyped_entities:


            # This should happen regardless!!!!
            remove_number_and_string_types(grammar_dictionary)

        return grammar_dictionary

    @staticmethod
    def is_global_rule(production_rule: str) -> bool:
        # we are checking -4 as is not a global rule if we
        # see the 0 in the a rule like 'value -> ["\'city_name0\'"]'
        if "value" in production_rule and production_rule[-4].isnumeric():
            return False
        return True


@Text2SqlWorld.register("linking")
class LinkingText2SqlWorld(Text2SqlWorld):
    """
    A World representation for any of the Text2Sql datasets which does entity linking
    by comparing words in the question to words in the database.

    Parameters
    ----------
    schema_path: ``str``
        A path to a schema file which we read into a dictionary
        representing the SQL tables in the dataset, the keys are the
        names of the tables that map to lists of the table's column names.
    cursor : ``Cursor``, required.
        An optional cursor for a database, which is used to add
        database values to the grammar.
    use_untyped_entities : ``bool``, optional (default = False)
        Whether or not to try to infer the types of prelinked variables.
        If not, they are added as untyped values to the grammar instead.
    """
    def __init__(self,
                 schema_path: str,
                 cursor: Cursor,
                 use_untyped_entities: bool = False) -> None:
        self.cursor = cursor
        self.schema = read_dataset_schema(schema_path)
        self.columns = {column.name: column for table in self.schema.values() for column in table}
        self.dataset_name = os.path.basename(schema_path).split("-")[0]
        self.use_untyped_entities = use_untyped_entities

        self.database_contents = database_to_dict(self.schema, self.cursor)

        # NOTE: This base dictionary should not be modified.
        self.base_grammar_dictionary = self._initialize_grammar_dictionary(deepcopy(GRAMMAR_DICTIONARY))

    def get_action_sequence_and_all_actions(self,
                                            query: List[str] = None,
                                            prelinked_entities: Dict[str, Dict[str, str]] = None) -> Tuple[List[str], List[str]]: # pylint: disable=line-too-long
        grammar_with_context = deepcopy(self.base_grammar_dictionary)

        if prelinked_entities is not None:
            raise ConfigurationError("The LinkingText2SqlWorld should not be passed prelinked entities. ")
        prelinked_entities = prelinked_entities or {}

        grammar = Grammar(format_grammar_string(grammar_with_context))

        valid_actions = initialize_valid_actions(grammar)
        all_actions = set()
        for action_list in valid_actions.values():
            all_actions.update(action_list)
        sorted_actions = sorted(all_actions)

        sql_visitor = SqlVisitor(grammar)
        try:
            action_sequence = sql_visitor.parse(" ".join(query)) if query else []
        except ParseError:
            action_sequence = None

        return action_sequence, sorted_actions, None

    def _initialize_grammar_dictionary(self, grammar_dictionary: Dict[str, List[str]]) -> Dict[str, List[str]]:
        # Add all the table and column names to the grammar.
        update_grammar_with_tables(grammar_dictionary, self.schema)

        # TODO Mark: Pretty sure we don't want to include all table values as actions.
        # Do something else instead. generate variables representing a pre-linked
        # value per column instead?
        if self.cursor is not None:
            # Now if we have strings in the table, we need to be able to
            # produce them, so we find all of the strings in the tables here
            # and create production rules from them. We only do this if
            # we haven't pre-linked entities, because if we have, we don't
            # need to be able to generate the values - just the placeholder
            # symbols which link to them.
            grammar_dictionary["number"] = []
            grammar_dictionary["string"] = []
            update_grammar_with_table_values(grammar_dictionary, self.schema, self.cursor)

        # Finally, update the grammar with global, non-variable values
        # found in the dataset, if present.
        update_grammar_with_global_values(grammar_dictionary, self.dataset_name)

        update_grammar_to_be_variable_free(grammar_dictionary)

        if self.use_untyped_entities:
            # This should happen regardless!!!!
            remove_number_and_string_types(grammar_dictionary)

        return grammar_dictionary

    def is_global_rule(self, production_rule: str) -> bool:
        return True

    @classmethod
    def from_params(cls, params: Params) -> 'LinkingText2SqlWorld':  # type: ignore

        schema_path = params.pop("schema_path")
        database_path = params.pop("database_path")
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
        use_untyped_entities = params.pop_bool("use_untyped_entities", True)

        return cls(schema_path=schema_path,
                   cursor=cursor,
                   use_untyped_entities=use_untyped_entities)
