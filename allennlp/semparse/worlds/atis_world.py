from typing import List, Dict, Tuple, Set, Callable
from copy import deepcopy
import numpy
from nltk import ngrams

from parsimonious.grammar import Grammar

from allennlp.semparse.contexts.atis_tables import * # pylint: disable=wildcard-import,unused-wildcard-import
from allennlp.semparse.contexts.atis_sql_table_context import AtisSqlTableContext, KEYWORDS
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor, format_action

from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

def get_strings_from_utterance(tokenized_utterance: List[Token]) -> Dict[str, List[int]]:
    """
    Based on the current utterance, return a dictionary where the keys are the strings in
    the database that map to lists of the token indices that they are linked to.
    """
    string_linking_scores: Dict[str, List[int]] = defaultdict(list)

    for index, token in enumerate(tokenized_utterance):
        for string in ATIS_TRIGGER_DICT.get(token.text.lower(), []):
            string_linking_scores[string].append(index)

    bigrams = ngrams([token.text for token in tokenized_utterance], 2)
    for index, bigram in enumerate(bigrams):
        for string in ATIS_TRIGGER_DICT.get(' '.join(bigram).lower(), []):
            string_linking_scores[string].extend([index,
                                                  index + 1])

    trigrams = ngrams([token.text for token in tokenized_utterance], 3)
    for index, trigram in enumerate(trigrams):
        for string in ATIS_TRIGGER_DICT.get(' '.join(trigram).lower(), []):
            string_linking_scores[string].extend([index,
                                                  index + 1,
                                                  index + 2])
    return string_linking_scores


class AtisWorld():
    """
    World representation for the Atis SQL domain. This class has a ``SqlTableContext`` which holds the base
    grammar, it then augments this grammar by constraining each column to the values that are allowed in it.

    Parameters
    ----------
    utterances: ``List[str]``
        A list of utterances in the interaction, the last element in this list is the
        current utterance that we are interested in.
    tokenizer: ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this tokenizer to tokenize the utterances.
    """

    database_file = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/atis/atis.db"
    sql_table_context = AtisSqlTableContext(ALL_TABLES,
                                            TABLES_WITH_STRINGS,
                                            database_file)

    def __init__(self,
                 utterances: List[str],
                 tokenizer: Tokenizer = None) -> None:

        self.utterances: List[str] = utterances
        self.tokenizer = tokenizer if tokenizer else WordTokenizer()
        self.tokenized_utterances = [self.tokenizer.tokenize(utterance) for utterance in self.utterances]
        self.linked_entities = self._get_linked_entities()
        self.dates = self._get_dates()

        self.valid_actions: Dict[str, List[str]] = self._update_valid_actions()
        entities, linking_scores = self._flatten_entities()
        # This has shape (num_entities, num_utterance_tokens).
        self.linking_scores: numpy.ndarray = linking_scores
        self.entities: List[str] = entities
        self.grammar_dictionary = self.update_grammar_dictionary()
        self.grammar_string: str = self.get_grammar_string()
        self.grammar_with_context: Grammar = Grammar(self.grammar_string)

    def get_valid_actions(self) -> Dict[str, List[str]]:
        return self.valid_actions

    def add_to_number_linking_scores(self,
                                     all_numbers: Set[str],
                                     number_linking_scores: Dict[str, Tuple[str, str, List[int]]],
                                     get_number_linking_dict: Callable[[str, List[Token]],
                                                                       Dict[str, List[int]]],
                                     current_tokenized_utterance: List[Token],
                                     nonterminal: str) -> None:
        """
        This is a helper method for adding different types of numbers (eg. starting time ranges) as entities.
        We first go through all utterances in the interaction and find the numbers of a certain type and add
        them to the set ``all_numbers``, which is initialized with default values. We want to add all numbers
        that occur in the interaction, and not just the current turn because the query could contain numbers
        that were triggered before the current turn. For each entity, we then check if it is triggered by tokens
        in the current utterance and construct the linking score.
        """
        number_linking_dict: Dict[str, List[int]] = {}
        for utterance, tokenized_utterance in zip(self.utterances, self.tokenized_utterances):
            number_linking_dict = get_number_linking_dict(utterance, tokenized_utterance)
            all_numbers.update(number_linking_dict.keys())
        all_numbers_list: List[str] = sorted(all_numbers, reverse=True)
        for number in all_numbers_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            # ``number_linking_dict`` is for the last utterance here. If the number was triggered
            # before the last utterance, then it will have linking scores of 0's.
            for token_index in number_linking_dict.get(number, []):
                if token_index < len(entity_linking):
                    entity_linking[token_index] = 1
            action = format_action(nonterminal, number, is_number=True, keywords_to_uppercase=KEYWORDS)
            number_linking_scores[action] = (nonterminal, number, entity_linking)


    def _get_linked_entities(self) -> Dict[str, Dict[str, Tuple[str, str, List[int]]]]:
        """
        This method gets entities from the current utterance finds which tokens they are linked to.
        The entities are divided into two main groups, ``numbers`` and ``strings``. We rely on these
        entities later for updating the valid actions and the grammar.
        """
        current_tokenized_utterance = [] if not self.tokenized_utterances \
                else self.tokenized_utterances[-1]

        # We generate a dictionary where the key is the type eg. ``number`` or ``string``.
        # The value is another dictionary where the key is the action and the value is a tuple
        # of the nonterminal, the string value and the linking score.
        entity_linking_scores: Dict[str, Dict[str, Tuple[str, str, List[int]]]] = {}

        number_linking_scores: Dict[str, Tuple[str, str, List[int]]] = {}
        string_linking_scores: Dict[str, Tuple[str, str, List[int]]] = {}

        # Get time range start
        self.add_to_number_linking_scores({'0'},
                                          number_linking_scores,
                                          get_time_range_start_from_utterance,
                                          current_tokenized_utterance,
                                          'time_range_start')

        self.add_to_number_linking_scores({"1200"},
                                          number_linking_scores,
                                          get_time_range_end_from_utterance,
                                          current_tokenized_utterance,
                                          'time_range_end')

        self.add_to_number_linking_scores({'0', '1'},
                                          number_linking_scores,
                                          get_numbers_from_utterance,
                                          current_tokenized_utterance,
                                          'number')

        # Add string linking dict.
        string_linking_dict: Dict[str, List[int]] = {}
        for tokenized_utterance in self.tokenized_utterances:
            string_linking_dict = get_strings_from_utterance(tokenized_utterance)
        strings_list = AtisWorld.sql_table_context.strings_list
        # We construct the linking scores for strings from the ``string_linking_dict`` here.
        for string in strings_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            # string_linking_dict has the strings and linking scores from the last utterance.
            # If the string is not in the last utterance, then the linking scores will be all 0.
            for token_index in string_linking_dict.get(string[1], []):
                entity_linking[token_index] = 1
            action = string[0]
            string_linking_scores[action] = (action.split(' -> ')[0], string[1], entity_linking)

        entity_linking_scores['number'] = number_linking_scores
        entity_linking_scores['string'] = string_linking_scores
        return entity_linking_scores

    def _update_valid_actions(self) -> Dict[str, List[str]]:
        valid_actions = deepcopy(self.sql_table_context.get_valid_actions())
        valid_actions['time_range_start'] = []
        valid_actions['time_range_end'] = []
        for action, value in self.linked_entities['number'].items():
            valid_actions[value[0]].append(action)

        for date in self.dates:
            for biexpr_rule in [f'biexpr -> ["date_day", ".", "year", binaryop, "{date.year}"]',
                                f'biexpr -> ["date_day", ".", "month_number", binaryop, "{date.month}"]',
                                f'biexpr -> ["date_day", ".", "day_number", binaryop, "{date.day}"]']:
                if biexpr_rule not in valid_actions:
                    valid_actions['biexpr'].append(biexpr_rule)

        valid_actions['ternaryexpr'] = \
                ['ternaryexpr -> [col_ref, "BETWEEN", time_range_start, "AND", time_range_end]',
                 'ternaryexpr -> [col_ref, "NOT", "BETWEEN", time_range_start, "AND", time_range_end]']

        return valid_actions

    def _get_dates(self):
        dates = []
        for tokenized_utterance in self.tokenized_utterances:
            dates.extend(get_date_from_utterance(tokenized_utterance))
        return dates

    def update_grammar_dictionary(self) -> Dict[str, List[str]]:
        """
        We modify the ``grammar_dictionary`` with additional constraints
        we want for the ATIS dataset. We then add numbers to the grammar dictionary. The strings in the
        database are already added in by the ``SqlTableContext``.
        """
        self.grammar_dictionary = deepcopy(self.sql_table_context.get_grammar_dictionary())
        if self.dates:
            year_binary_expression = f'("date_day" ws "." ws "year" ws binaryop ws "{self.dates[0].year}")'
            self.grammar_dictionary['biexpr'].append(year_binary_expression)

            for date in self.dates:
                month_day_binary_expressions = \
                        [f'("date_day" ws "." ws "month_number" ws binaryop ws "{date.month}")',
                         f'("date_day" ws "." ws "day_number" ws binaryop ws "{date.day}")']
                self.grammar_dictionary['biexpr'].extend(month_day_binary_expressions)


        self.grammar_dictionary['ternaryexpr'] = \
                ['(col_ref ws "not" ws "BETWEEN" ws time_range_start ws "AND" ws time_range_end ws)',
                 '(col_ref ws "NOT" ws "BETWEEN" ws time_range_start  ws "AND" ws time_range_end ws)',
                 '(col_ref ws "BETWEEN" ws time_range_start ws "AND" ws time_range_end ws)']

        # We need to add the numbers, starting, ending time ranges to the grammar.
        numbers = sorted([value[1] for key, value in self.linked_entities['number'].items()
                          if value[0] == 'number'], reverse=True)
        self.grammar_dictionary['number'] = [f'"{number}"' for number in numbers]

        time_range_start = sorted([value[1] for key, value in self.linked_entities['number'].items()
                                   if value[0] == 'time_range_start'], reverse=True)
        self.grammar_dictionary['time_range_start'] = [f'"{time}"' for time in time_range_start]

        time_range_end = sorted([value[1] for key, value in self.linked_entities['number'].items()
                                 if value[0] == 'time_range_end'], reverse=True)
        self.grammar_dictionary['time_range_end'] = [f'"{time}"' for time in time_range_end]
        return self.grammar_dictionary

    def get_grammar_string(self) -> str:
        """
        Generate a string that can be used to instantiate a ``Grammar`` object. The string is a sequence
        of rules that define the grammar.
        """
        return '\n'.join([f"{nonterminal} = {' / '.join(right_hand_side)}"
                          for nonterminal, right_hand_side in self.grammar_dictionary.items()])

    def get_action_sequence(self, query: str) -> List[str]:
        sql_visitor = SqlVisitor(self.grammar_with_context, keywords_to_uppercase=KEYWORDS)
        if query:
            action_sequence = sql_visitor.parse(query)
            return action_sequence
        return []

    def all_possible_actions(self) -> List[str]:
        """
        Return a sorted list of strings representing all possible actions
        of the form: nonterminal -> [right_hand_side]
        """
        all_actions = set()
        for _, action_list in self.valid_actions.items():
            for action in action_list:
                all_actions.add(action)
        return sorted(all_actions)

    def _flatten_entities(self) -> Tuple[List[str], numpy.ndarray]:
        """
        When we first get the entities and the linking scores in ``_get_linked_entities``
        we represent as dictionaries for easier updates to the grammar and valid actions.
        In this method, we flatten them for the model so that the entities are represented as
        a list, and the linking scores are a 2D numpy array of shape (num_entities, num_utterance_tokens).
        """
        entities = []
        linking_scores = []
        for entity in sorted(self.linked_entities['number']):
            entities.append(entity)
            linking_scores.append(self.linked_entities['number'][entity][2])

        for entity in sorted(self.linked_entities['string']):
            entities.append(entity)
            linking_scores.append(self.linked_entities['string'][entity][2])
        return entities, numpy.array(linking_scores)


    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return all([self.valid_actions == other.valid_actions,
                        numpy.array_equal(self.linking_scores, other.linking_scores),
                        self.utterances == other.utterances,
                        self.grammar_string == other.grammar_string])
        return False
