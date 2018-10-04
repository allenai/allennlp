from typing import List, Dict, Tuple, Set, Callable
from copy import copy, deepcopy
import numpy
from nltk import ngrams

from parsimonious.grammar import Grammar
from parsimonious.expressions import Expression, OneOf, Sequence, Literal

from allennlp.semparse.contexts.atis_tables import * # pylint: disable=wildcard-import,unused-wildcard-import
from allennlp.semparse.contexts.atis_sql_table_context import AtisSqlTableContext, KEYWORDS
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor, format_action, initialize_valid_actions

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
    sql_table_context = None

    def __init__(self,
                 utterances: List[str],
                 tokenizer: Tokenizer = None) -> None:
        if AtisWorld.sql_table_context is None:
            AtisWorld.sql_table_context = AtisSqlTableContext(ALL_TABLES,
                                                              TABLES_WITH_STRINGS,
                                                              AtisWorld.database_file)
        self.utterances: List[str] = deepcopy(utterances)
        self.tokenizer = tokenizer if tokenizer else WordTokenizer()
        self.tokenized_utterances = [self.tokenizer.tokenize(utterance) for utterance in self.utterances]
        self.linked_entities = self._get_linked_entities()
        self.dates = self._get_dates()

        entities, linking_scores = self._flatten_entities()
        # This has shape (num_entities, num_utterance_tokens).
        self.linking_scores: numpy.ndarray = linking_scores
        self.entities: List[str] = entities
        self.grammar: Grammar = self._update_grammar()
        self.valid_actions = initialize_valid_actions(self.grammar,
                                                      KEYWORDS)

    def _update_grammar(self):
        """
        We create a new ``Grammar`` object from the one in ``AtisSqlTableContext``, that also
        has the new entities that are extracted from the utterance. Stitching together the expressions
        to form the grammar is a little tedious here, but it is worth it because we don't have to create
        a new grammar from scratch. Creating a new grammar is expensive because we have many production
        rules that have all database values in the column on the right hand side. We update the expressions
        bottom up, since the higher level expressions may refer to the lower level ones. For example, the
        ternary expression will refer to the start and end times.
        """

        # This will give us a shallow copy. We have to be careful here because the ``Grammar`` object
        # contains ``Expression`` objects that have tuples containing the members of that expression.
        # We have to create new sub-expression objects so that original grammar is not mutated.
        new_grammar = copy(AtisWorld.sql_table_context.grammar)

        numbers = self._get_numeric_database_values('number')
        number_literals = [Literal(number) for number in numbers]
        new_grammar['number'] = OneOf(*number_literals, name='number')
        self._update_expression_reference(new_grammar, 'pos_value', 'number')

        time_range_start = self._get_numeric_database_values('time_range_start')
        time_range_start_literals = [Literal(time) for time in time_range_start]
        new_grammar['time_range_start'] = OneOf(*time_range_start_literals, name='time_range_start')

        time_range_end = self._get_numeric_database_values('time_range_end')
        time_range_end_literals = [Literal(time) for time in time_range_end]
        new_grammar['time_range_end'] = OneOf(*time_range_end_literals, name='time_range_end')

        fare_round_trip_cost= self._get_numeric_database_values('fare_round_trip_cost')
        fare_round_trip_cost_literals = [Literal(fare) for fare in fare_round_trip_cost]
        new_grammar['fare_round_trip_cost'] = OneOf(*fare_round_trip_cost_literals, name='fare_round_trip_cost')

        fare_one_direction_cost= self._get_numeric_database_values('fare_one_direction_cost')
        fare_one_direction_cost_literals = [Literal(fare) for fare in fare_one_direction_cost]
        new_grammar['fare_one_direction_cost'] = OneOf(*fare_one_direction_cost_literals, name='fare_one_direction_cost')

        ternary_expressions = [self._get_sequence_with_spacing(new_grammar,
                                                               [new_grammar['col_ref'],
                                                                Literal('BETWEEN'),
                                                                new_grammar['time_range_start'],
                                                                Literal(f'AND'),
                                                                new_grammar['time_range_end']]),
                               self._get_sequence_with_spacing(new_grammar,
                                                               [new_grammar['col_ref'],
                                                                Literal('NOT'),
                                                                Literal('BETWEEN'),
                                                                new_grammar['time_range_start'],
                                                                Literal(f'AND'),
                                                                new_grammar['time_range_end']]),
                               self._get_sequence_with_spacing(new_grammar,
                                                               [new_grammar['col_ref'],
                                                                Literal('not'),
                                                                Literal('BETWEEN'),
                                                                new_grammar['time_range_start'],
                                                                Literal(f'AND'),
                                                                new_grammar['time_range_end']])]

        new_grammar['ternaryexpr'] = OneOf(*ternary_expressions, name='ternaryexpr')
        self._update_expression_reference(new_grammar, 'condition', 'ternaryexpr')

        new_binary_expressions = []

        fare_round_trip_cost_expression = \
                    self._get_sequence_with_spacing(new_grammar,
                                                    [Literal('fare'),
                                                     Literal('.'),
                                                     Literal('round_trip_cost'),
                                                     new_grammar['binaryop'],
                                                     new_grammar['fare_round_trip_cost']])

        new_binary_expressions.append(fare_round_trip_cost_expression)

        fare_one_direction_cost_expression = \
                    self._get_sequence_with_spacing(new_grammar,
                                                    [Literal('fare'),
                                                     Literal('.'),
                                                     Literal('one_direction_cost'),
                                                     new_grammar['binaryop'],
                                                     new_grammar['fare_one_direction_cost']])

        new_binary_expressions.append(fare_one_direction_cost_expression)

        if self.dates:
            year_binary_expression = self._get_sequence_with_spacing(new_grammar,
                                                                     [Literal('date_day'),
                                                                      Literal('.'),
                                                                      Literal('year'),
                                                                      new_grammar['binaryop'],
                                                                      Literal(f'{self.dates[0].year}')])
            new_binary_expressions.append(year_binary_expression)
            for date in self.dates:
                month_binary_expression = self._get_sequence_with_spacing(new_grammar,
                                                                          [Literal('date_day'),
                                                                           Literal('.'),
                                                                           Literal('month_number'),
                                                                           new_grammar['binaryop'],
                                                                           Literal(f'{date.month}')])
                day_binary_expression = self._get_sequence_with_spacing(new_grammar,
                                                                        [Literal('date_day'),
                                                                         Literal('.'),
                                                                         Literal('day_number'),
                                                                         new_grammar['binaryop'],
                                                                         Literal(f'{date.day}')])
                new_binary_expressions.extend([month_binary_expression,
                                               day_binary_expression])
            
            
            
        new_binary_expressions = new_grammar['biexpr'].members + tuple(new_binary_expressions)
        new_grammar['biexpr'] = OneOf(*new_binary_expressions, name='biexpr')
        self._update_expression_reference(new_grammar, 'condition', 'biexpr')

        return new_grammar

    def _get_numeric_database_values(self,
                                     nonterminal: str) -> List[str]:
        return sorted([value[1] for key, value in self.linked_entities['number'].items()
                       if value[0] == nonterminal], reverse=True)

    def _update_expression_reference(self, # pylint: disable=no-self-use
                                     grammar: Grammar,
                                     parent_expression_nonterminal: str,
                                     child_expression_nonterminal: str) -> None:
        """
        When we add a new expression, there may be other expressions that refer to
        it, and we need to update those to point to the new expression.
        """
        grammar[parent_expression_nonterminal].members = \
                [member if member.name != child_expression_nonterminal
                 else grammar[child_expression_nonterminal]
                 for member in grammar[parent_expression_nonterminal].members]

    def _get_sequence_with_spacing(self, # pylint: disable=no-self-use
                                   new_grammar,
                                   expressions: List[Expression],
                                   name: str = '') -> Sequence:
        """
        This is a helper method for generating sequences, since we often want a list of expressions
        with whitespaces between them.
        """
        expressions = [subexpression
                       for expression in expressions
                       for subexpression in (expression, new_grammar['ws'])]
        return Sequence(*expressions, name=name)

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

        self.add_to_number_linking_scores({'1200'},
                                          number_linking_scores,
                                          get_time_range_end_from_utterance,
                                          current_tokenized_utterance,
                                          'time_range_end')

        self.add_to_number_linking_scores({'0', '1'},
                                          number_linking_scores,
                                          get_numbers_from_utterance,
                                          current_tokenized_utterance,
                                          'number')

        self.add_to_number_linking_scores({'0'},
                                          number_linking_scores,
                                          get_costs_from_utterance,
                                          current_tokenized_utterance,
                                          'fare_round_trip_cost')

        self.add_to_number_linking_scores({'0'},
                                          number_linking_scores,
                                          get_costs_from_utterance,
                                          current_tokenized_utterance,
                                          'fare_one_direction_cost')

        # Add string linking dict.
        string_linking_dict: Dict[str, List[int]] = {}
        for tokenized_utterance in self.tokenized_utterances:
            string_linking_dict = get_strings_from_utterance(tokenized_utterance)
        strings_list = AtisWorld.sql_table_context.strings_list
        strings_list.append(('flight_airline_code_string -> ["\'EA\'"]', 'EA'))
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

    def _get_dates(self):
        dates = [] 
        for tokenized_utterance in self.tokenized_utterances:
            dates.extend(get_date_from_utterance(tokenized_utterance))
        return dates

    def get_action_sequence(self, query: str) -> List[str]:
        sql_visitor = SqlVisitor(self.grammar, keywords_to_uppercase=KEYWORDS)
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
                        self.utterances == other.utterances])
        return False
