from copy import deepcopy
from typing import List, Dict, Tuple
import numpy
from pprint import pprint
import sqlite3
from nltk import ngrams

from parsimonious.grammar import Grammar

from allennlp.semparse.contexts.atis_tables import * # pylint: disable=wildcard-import,unused-wildcard-import
from allennlp.semparse.contexts.sql_table_context import \
        SqlTableContext, SqlVisitor, generate_one_of_string, format_action

from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

def get_strings_from_utterance(tokenized_utterance: List[Token]) -> Dict[str, List[int]]:
    """
    Based on the current utterance, return a dictionary where the keys are the strings in the utterance
    that map to lists of the token indices that they are linked to.
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
    grammars, it then augments this grammar with the entities that are detected from utterances.

    Parameters
    ----------
    utterances: ``List[str]``
        A list of utterances in the interaction, the last element in this list is the
        current utterance that we are interested in.
    tokenizer: ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this tokenizer to tokenize the utterances.
    database_directory: ``str``, optional
        We pass the location of the database directory to ``SqlTableContext`` to get the allowed strings in
        the grammar.
    """

    def __init__(self,
                 utterances: List[str],
                 tokenizer: Tokenizer = None,
                 database_directory: str = None) -> None:

        self.all_tables = ALL_TABLES
        self.tables_with_strings = TABLES_WITH_STRINGS 
        if database_directory:
            self.database_directory = database_directory
            self.connection = sqlite3.connect(database_directory)
            self.cursor = self.connection.cursor()


        self.sql_table_context = SqlTableContext(ALL_TABLES,
                                                 TABLES_WITH_STRINGS,
                                                 database_directory,
                                                 utterances=utterances) if database_directory else None

        self.utterances: List[str] = utterances
        self.tokenizer = tokenizer if tokenizer else WordTokenizer()
        self.tokenized_utterances = [self.tokenizer.tokenize(utterance) for utterance in self.utterances]
        valid_actions, linking_scores = self.init_all_valid_actions()
        self.valid_actions: Dict[str, List[str]] = valid_actions
        self.entities = self.get_entities()


        # This has shape (num_entities, num_utterance_tokens).
        self.linking_scores: numpy.ndarray = linking_scores
        self.grammar_str: str = self.get_grammar_str()
        self.grammar_with_context: Grammar = Grammar(self.grammar_str)
        all_possible_actions = self.all_possible_actions()

        if database_directory:
            self.connection.close()
        

    def get_valid_actions(self) -> Dict[str, List[str]]:
        return self.valid_actions

    def init_all_valid_actions(self) -> Tuple[Dict[str, List[str]], numpy.ndarray]:
        valid_actions = deepcopy(self.sql_table_context.valid_actions)
        linking_scores = []

        current_tokenized_utterance = [] if not self.tokenized_utterances \
                else self.tokenized_utterances[-1]
        
        # Get time range start
        time_range_start = {"0"}
        time_range_start_linking_dict: Dict[str, List[int]] = {}

        for utterance, tokenized_utterance in zip(self.utterances, self.tokenized_utterances):
            time_range_start_linking_dict = get_time_range_start_from_utterance(utterance, tokenized_utterance)
            time_range_start.update(time_range_start_linking_dict.keys())
        time_range_start_list: List[str] = sorted(time_range_start, reverse=True)

        time_range_start_linking_scores = []
        for time in time_range_start_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            for token_index in time_range_start_linking_dict.get(time, []):
                entity_linking[token_index] = 1
            time_range_start_linking_scores.append(entity_linking)
        linking_scores.extend(time_range_start_linking_scores)

        for time in list(time_range_start_list):
            action = format_action('time_range_start', time, is_number=True)
            valid_actions['time_range_start'].append(action)

        # Get time range end
        time_range_end = {"1200"}
        time_range_end_linking_dict: Dict[str, List[int]] = {}

        for utterance, tokenized_utterance in zip(self.utterances, self.tokenized_utterances):
            time_range_end_linking_dict = get_time_range_end_from_utterance(utterance, tokenized_utterance)
            time_range_end.update(time_range_end_linking_dict.keys())
        time_range_end_list: List[str] = sorted(time_range_end, reverse=True)

        time_range_end_linking_scores = []
        for time in time_range_end_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            for token_index in time_range_end_linking_dict.get(time, []):
                entity_linking[token_index] = 1
            time_range_end_linking_scores.append(entity_linking)
        linking_scores.extend(time_range_end_linking_scores)

        for time in list(time_range_end_list):
            action = format_action('time_range_end', time, is_number=True)
            valid_actions['time_range_end'].append(action)
        
        
        numbers = {'0', '1'}
        number_linking_dict: Dict[str, List[int]] = {}

        for utterance, tokenized_utterance in zip(self.utterances, self.tokenized_utterances):
            number_linking_dict = get_numbers_from_utterance(utterance, tokenized_utterance)
            numbers.update(number_linking_dict.keys())
        numbers_list: List[str] = sorted(numbers, reverse=True)

        # We construct the linking scores for numbers from the ``number_linking_dict`` here.
        number_linking_scores = []
        for number in numbers_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            # number_linking_scores has the numbers and linking scores from the last utterance.
            # If the number is not in the last utterance, then the linking scores will be all 0.
            for token_index in number_linking_dict.get(number, []):
                if token_index <= len(entity_linking) - 1:
                    entity_linking[token_index] = 1
            number_linking_scores.append(entity_linking)
        linking_scores.extend(number_linking_scores)

        for number in list(numbers_list):
            action = format_action('number', number, is_number=True)
            valid_actions['number'].append(action)
        
        # add string linking dict
        string_linking_dict: Dict[str, List[int]] = {}
        for tokenized_utterance in self.tokenized_utterances:
            string_linking_dict = get_strings_from_utterance(tokenized_utterance)
            # strings.update(string_linking_dict.keys())
        # strings_list: List[str] = sorted(strings, reverse=True)
        strings_list = []

        if self.tables_with_strings:
            for table, columns in self.tables_with_strings.items():
                for column in columns:
                    self.cursor.execute(f'SELECT DISTINCT {table} . {column} FROM {table}')
                    strings_list.extend([(format_action(f"{table}_{column}_string", str(row[0]), is_string=True), str(row[0]))
                                                for row in self.cursor.fetchall()])
        strings_list = sorted(strings_list, key=lambda string_tuple: string_tuple[0], reverse=True)
        # We construct the linking scores for strings from the ``string_linking_dict`` here.

        string_linking_scores = []

        for string in strings_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            # string_linking_dict has the strings and linking scores from the last utterance.
            # If the string is not in the last utterance, then the linking scores will be all 0.
            for token_index in string_linking_dict.get(string[1], []):
                entity_linking[token_index] = 1

            string_linking_scores.append(entity_linking)
        linking_scores.extend(string_linking_scores)

        return valid_actions, numpy.array(linking_scores)

    def get_grammar_str(self) -> str:
        """
        Generate a string that can be used to instantiate a ``Grammar`` object. The string is a sequence of
        rules that define the grammar.
        """
        grammar_str_with_context = self.sql_table_context.grammar_str
        grammar_str_lines = [line for line in grammar_str_with_context.split('\n') if line]

        dates = []
        for tokenized_utterance in self.tokenized_utterances:
            dates.extend(get_date_from_utterance(tokenized_utterance))
        if dates:
            grammar_str_lines[-1] += f' / ("date_day" ws "." ws "year" ws binaryop ws "{dates[0].year}")'
            for date in dates:
                grammar_str_lines[-1] += f' / ("date_day" ws "." ws "month_number" ws binaryop ws "{date.month}")' \
                                         f' / ("date_day" ws "." ws "day_number" ws binaryop ws "{date.day}")'
                for biexpr_rule in [f'biexpr -> ["date_day", ".", "year", binaryop, "{date.year}"]',
                                    f'biexpr -> ["date_day", ".", "month_number", binaryop, "{date.month}"]',
                                    f'biexpr -> ["date_day", ".", "day_number", binaryop, "{date.day}"]']:
                    if biexpr_rule not in self.valid_actions:
                        self.valid_actions['biexpr'].append(biexpr_rule)
        # add ternary expression 
        grammar_str_lines.append('ternaryexpr = (col_ref ws "not" ws "BETWEEN" ws time_range_start ws "AND" ws time_range_end ws) / (col_ref ws "NOT" ws "BETWEEN" ws time_range_start  ws "AND" ws time_range_end ws) /  \
                          (col_ref ws "BETWEEN" ws time_range_start ws "AND" ws time_range_end ws)\n')

        self.valid_actions['ternaryexpr'] = ['ternaryexpr -> [col_ref, "BETWEEN", time_range_start, "AND", time_range_end]', 'ternaryexpr -> [col_ref, "NOT", "BETWEEN", time_range_start, "AND", time_range_end]']

        grammar_str_with_context = "\n".join(grammar_str_lines)

        numbers = [number.split(" -> ")[1].lstrip('["').rstrip('"]') for \
                   number in sorted(self.valid_actions['number'], reverse=True)]

        grammar_str_with_context += generate_one_of_string("number", numbers)

        time_range_start = [time.split(" -> ")[1].lstrip('["').rstrip('"]') for \
                   time in sorted(self.valid_actions['time_range_start'], reverse=True)]
        grammar_str_with_context += generate_one_of_string("time_range_start", time_range_start)

        time_range_end = [time.split(" -> ")[1].lstrip('["').rstrip('"]') for \
                   time in sorted(self.valid_actions['time_range_end'], reverse=True)]
        grammar_str_with_context += generate_one_of_string("time_range_end", time_range_end)
        return grammar_str_with_context


    def get_action_sequence(self, query: str) -> List[str]:
        sql_visitor = SqlVisitor(self.grammar_with_context)
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

    def get_entities(self) -> List[str]:
        time_range_start = self.valid_actions['time_range_start']
        time_range_end= self.valid_actions['time_range_end']
        numbers = self.valid_actions['number']
        strings = []
        for _, action_list in self.valid_actions.items():
            for action in action_list:
                lhs, _ = action.split(' ->')
                if lhs.endswith('string'):
                    strings.append(action)
        return sorted(time_range_start, reverse=True) + sorted(time_range_end, reverse=True)+ \
               sorted(numbers, reverse=True) + sorted(strings, reverse=True)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return all([self.valid_actions == other.valid_actions,
                        numpy.array_equal(self.linking_scores, other.linking_scores),
                        self.utterances == other.utterances,
                        self.grammar_str == other.grammar_str])
        return False



