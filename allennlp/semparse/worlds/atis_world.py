from copy import deepcopy
from typing import List, Dict

from parsimonious.grammar import Grammar

from allennlp.semparse.contexts.atis_tables import * # pylint: disable=wildcard-import,unused-wildcard-import
from allennlp.semparse.contexts.sql_table_context import \
        SqlTableContext, SqlVisitor, generate_one_of_string, format_action

from allennlp.data.tokenizers import WordTokenizer


class AtisWorld():
    """
    World representation for the Atis SQL domain. This class has a ``SqlTableContext`` which holds the base
    grammars, it then augments this grammar with the entities that are detected from utterances.

    Parameters
    ----------
    utterances: ``List[str]``
        A list of utterances in the interaction, the last element in this list is the
        current utterance that we are interested in.
    """
    sql_table_context = SqlTableContext(TABLES)

    def __init__(self, utterances: List[str], tokenizer=None) -> None:
        self.utterances: List[str] = utterances
        self.tokenizer = tokenizer if tokenizer else WordTokenizer()
        self.tokenized_utterances = [self.tokenizer.tokenize(utterance) for utterance in self.utterances]
        self.valid_actions: Dict[str, List[str]] = self.init_all_valid_actions()
        self.grammar_str: str = self.get_grammar_str()
        self.grammar_with_context: Grammar = Grammar(self.grammar_str)

    def get_valid_actions(self) -> Dict[str, List[str]]:
        return self.valid_actions

    def init_all_valid_actions(self) -> Dict[str, List[str]]:
        """
        We initialize the world's valid actions with that of the context. This means that the strings
        and numbers that were valid earlier in the interaction are also valid. We then add new valid strings
        and numbers from the current utterance.
        """
        valid_actions = deepcopy(self.sql_table_context.valid_actions)
        for string in self.get_strings_from_utterance():
            action = format_action('string', string)
            if action not in valid_actions['string']:
                valid_actions['string'].append(action)

        numbers = ['0', '1']
        for utterance in self.utterances:
            numbers.extend(get_numbers_from_utterance(utterance))
            for number in numbers:
                action = format_action('number', number)
                if action not in valid_actions['number']:
                    valid_actions['number'].append(action)

        return valid_actions

    def get_grammar_str(self) -> str:
        """
        Generate a string that can be used to instantiate a ``Grammar`` object. The string is a sequence of
        rules that define the grammar.
        """
        grammar_str_with_context = self.sql_table_context.grammar_str
        numbers = [number.split(" -> ")[1].lstrip('["').rstrip('"]') for \
                   number in sorted(self.valid_actions['number'], reverse=True)]
        strings = [string .split(" -> ")[1].lstrip('["').rstrip('"]') for \
                   string in sorted(self.valid_actions['string'], reverse=True)]

        grammar_str_with_context += generate_one_of_string("number", numbers)
        grammar_str_with_context += generate_one_of_string("string", strings)
        return grammar_str_with_context


    def get_strings_from_utterance(self) -> List[str]:
        """
        Based on the current utterance, return a list of valid strings that should be added.
        """
        strings: List[str] = []

        for tokenized_utterance in self.tokenized_utterances:
            for first_token, second_token in zip(tokenized_utterance, tokenized_utterance[1:]):
                strings.extend(ATIS_TRIGGER_DICT.get(first_token.text.lower(), []))
                bigram = f"{first_token.text} {second_token.text}".lower()
                strings.extend(ATIS_TRIGGER_DICT.get(bigram, []))
            strings.extend(ATIS_TRIGGER_DICT.get(tokenized_utterance[-1].text.lower(), []))
            date = get_date_from_utterance(tokenized_utterance)
            if date:
                strings.extend(DAY_OF_WEEK_INDEX.get(date.weekday(), []))

        return strings

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
