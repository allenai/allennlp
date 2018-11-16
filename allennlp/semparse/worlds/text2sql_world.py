from typing import List, Tuple, Dict
from copy import deepcopy
import json
import os

from parsimonious import Grammar
from parsimonious.exceptions import ParseError
from nltk import ngrams, bigrams

from allennlp.common.registrable import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor
from allennlp.semparse.contexts.sql_context_utils import format_grammar_string, initialize_valid_actions
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import read_dataset_schema
from allennlp.semparse.contexts.text2sql_table_context import GRAMMAR_DICTIONARY
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_tables
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_global_values
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_to_be_variable_free
from allennlp.semparse.contexts.text2sql_table_context import remove_number_and_string_types
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_values_with_variables
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_typed_variables
from allennlp.semparse.contexts.text2sql_table_context import update_grammar_with_linked_typed_variables
from allennlp.data.tokenizers import Token

class Text2SqlWorld(Registrable):

    def get_action_sequence_and_all_actions(self,
                                            query: List[str] = None,
                                            prelinked_entities: Dict[str, Dict[str, str]] = None) -> Tuple[List[str], List[str]]: # pylint: disable=line-too-long
        raise NotImplementedError

    def is_global_rule(self, nonterminal: str) -> bool:

        raise NotImplementedError


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
                 use_untyped_entities: bool = False,
                 link_entities_to_actions: bool = False) -> None:
        self.schema = read_dataset_schema(schema_path)
        self.typed_variable_nonterminals = {f"{table_name}_{column.name}_value"
                                            for table_name, table in self.schema.items()
                                            for column in table}

        self.dataset_name = os.path.basename(schema_path).split("-")[0]
        self.use_untyped_entities = use_untyped_entities
        self.link_entities_to_actions = link_entities_to_actions

        if link_entities_to_actions and use_untyped_entities:
            raise ConfigurationError("To link entities to actions, you cannot use untyped entities.")

        # NOTE: This base dictionary should not be modified.
        self.base_grammar_dictionary = self._initialize_grammar_dictionary(deepcopy(GRAMMAR_DICTIONARY))

    def get_action_sequence_and_all_actions(self,
                                            query: List[str] = None,
                                            prelinked_entities: Dict[str, Dict[str, str]] = None) -> Tuple[List[str], List[str]]: # pylint: disable=line-too-long
        grammar_with_context = deepcopy(self.base_grammar_dictionary)


        prelinked_entities = prelinked_entities or {}
        if self.use_untyped_entities:
            update_grammar_values_with_variables(grammar_with_context, prelinked_entities)
        elif self.link_entities_to_actions:
            update_grammar_with_linked_typed_variables(grammar_with_context, prelinked_entities, self.dataset_name)
        else:
            update_grammar_with_typed_variables(grammar_with_context, prelinked_entities, self.dataset_name)

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
        update_grammar_with_tables(grammar_dictionary, self.schema, constrained=not self.use_untyped_entities)

        # Finally, update the grammar with global, non-variable values
        # found in the dataset, if present.
        update_grammar_with_global_values(grammar_dictionary, self.dataset_name)

        update_grammar_to_be_variable_free(grammar_dictionary)

        remove_number_and_string_types(grammar_dictionary)

        return grammar_dictionary

    def is_global_rule(self, nonterminal: str) -> bool:
        if self.link_entities_to_actions:
            if nonterminal in self.typed_variable_nonterminals:
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
    database_dict_path : ``str``, required.
        An optional path to a json file containing the values to be linked to.
    use_untyped_entities : ``bool``, optional (default = False)
        Whether or not to try to infer the types of prelinked variables.
        If not, they are added as untyped values to the grammar instead.
    """
    def __init__(self,
                 schema_path: str,
                 database_dict_path: str = None,
                 use_untyped_entities: bool = False) -> None:
        self.schema = read_dataset_schema(schema_path)
        self.typed_variable_nonterminals = {f"{table_name}_{column.name}_value"
                                            for table_name, table in self.schema.items()
                                            for column in table}
        self.dataset_name = os.path.basename(schema_path).split("-")[0]
        self.use_untyped_entities = use_untyped_entities

        self.database_contents = json.load(open(database_dict_path)) if database_dict_path is not None else None

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
        update_grammar_with_tables(grammar_dictionary, self.schema, constrained=not self.use_untyped_entities)

        # TODO Mark: Pretty sure we don't want to include all table values as actions.
        # Do something else instead. generate variables representing a pre-linked
        # value per column instead?

        # Finally, update the grammar with global, non-variable values
        # found in the dataset, if present.
        update_grammar_with_global_values(grammar_dictionary, self.dataset_name)

        update_grammar_to_be_variable_free(grammar_dictionary)

        remove_number_and_string_types(grammar_dictionary)

        return grammar_dictionary

    def is_global_rule(self, nonterminal: str) -> bool:
        if nonterminal in self.typed_variable_nonterminals:
            return False
        return True
