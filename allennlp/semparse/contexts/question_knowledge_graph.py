from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Union, Set

from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph

DEFAULT_NUMBERS = []
NUMBER_CHARACTERS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-'}
MONTH_NUMBERS = {
        'january': 1,
        'jan': 1,
        'february': 2,
        'feb': 2,
        'march': 3,
        'mar': 3,
        'april': 4,
        'apr': 4,
        'may': 5,
        'june': 6,
        'jun': 6,
        'july': 7,
        'jul': 7,
        'august': 8,
        'aug': 8,
        'september': 9,
        'sep': 9,
        'october': 10,
        'oct': 10,
        'november': 11,
        'nov': 11,
        'december': 12,
        'dec': 12,
        }
ORDER_OF_MAGNITUDE_WORDS = {'hundred': 100, 'thousand': 1000, 'million': 1000000}

NUMBER_WORDS = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'first': 1,
        'second': 2,
        'third': 3,
        'fourth': 4,
        'fifth': 5,
        'sixth': 6,
        'seventh': 7,
        'eighth': 8,
        'ninth': 9,
        'tenth': 10,
        **MONTH_NUMBERS,
        }

MULTIPLICATIVE_NUMBER_WORDS = {
        'half': 0.5,
        'once': 1,
        'twice': 2,
        'thrice': 3
}

MULTIPLIERS = {
        'single': 1,
        'double': 2,
        'triple': 3
}

COIN_VALUES = {
        'dime': 0.1,
        'dimes': 0.1,
        'nickel': 0.05,
        'nickels': 0.05
}

LEG_COUNT = {
        'chicken': 2,
        'chickens': 2,
        'calf': 4,
        'calves': 4,
        'cow': 4,
        'cows': 4,
        'duck': 2,
        'ducks': 2,
        'duckling': 2,
        'ducklings': 2,
        'goat': 4,
        'goats': 4,
        'goose': 2,
        'geese': 2,
        'hen': 2,
        'hens': 2,
        'horse': 4,
        'horses': 4,
        'lamb': 4,
        'lambs': 4,
        'sheep': 4,
        'rabbit': 4,
        'rabbits': 4,
        'rooster': 2,
        'roosters': 2,
        'turkey': 2,
        'turkeys': 2,
        'bunny': 4,
        'bunnies': 4,
        'pig': 4,
        'pigs': 4,
        'sow': 4,
        'sows': 4,
        'piglet': 4,
        'piglets': 4,
        'dog': 4,
        'dogs': 4,
        'cat': 4,
        'cats': 4
}

class QuestionKnowledgeGraph(KnowledgeGraph):
    """
    A ``QuestionKnowledgeGraph`` represents the linkable entities in a question.
    The linkable entities from the question are the numbers in the question.  We use the
    question to define our space of allowable numbers, because there are infinitely many numbers
    that we could include in our action space, and we really don't want to do that.

    We represent numbers as standalone nodes in the graph, without any neighbors.

    """
    @classmethod
    def read(cls, question: List[Token]) -> 'QuestionKnowledgeGraph':
        entity_text: Dict[str, str] = {}
        neighbors: DefaultDict[str, List[str]] = defaultdict(list)

        # Getting number entities first.  Number entities don't have any neighbors, and their
        # "entity text" is the text from the question that evoked the number.
        for number, number_text in cls._get_numbers_from_tokens(question):
            entity_text[number] = number_text
            neighbors[number] = []
        for default_number in DEFAULT_NUMBERS:
            if default_number not in neighbors:
                neighbors[default_number] = []
                entity_text[default_number] = default_number

        return cls(set(neighbors.keys()), dict(neighbors), entity_text)

    @staticmethod
    def _get_numbers_from_tokens(tokens: List[Token]) -> List[Tuple[str, str]]:
        """
        Finds numbers in the input tokens and returns them as strings.  We do some simple heuristic
        number recognition, finding ordinals and cardinals expressed as text ("one", "first",
        etc.), as well as numerals ("7th", "3rd"), months (mapping "july" to 7), and units
        ("1ghz").

        We also handle year ranges expressed as decade or centuries ("1800s" or "1950s"), adding
        the endpoints of the range as possible numbers to generate.

        We return a list of tuples, where each tuple is the (number_string, token_text) for a
        number found in the input tokens.
        """
        numbers = []
        #print("Tokens: ", tokens)
        for i, token in enumerate(tokens):
            number: Union[int, float] = None
            token_text = token.text
            text = token.text.replace(',', '').lower()
            if len(text) > 0 and text[0] != "-":
                text = text.split("-")[0]
            if text in NUMBER_WORDS:
                number = NUMBER_WORDS[text]

            if text in MULTIPLICATIVE_NUMBER_WORDS:
                number = MULTIPLICATIVE_NUMBER_WORDS[text]

            if text in MULTIPLIERS:
                number = MULTIPLIERS[text]

            if text in COIN_VALUES:
                number = COIN_VALUES[text]

            if text in LEG_COUNT:
                number = LEG_COUNT[text]

            magnitude = 1
            if i < len(tokens) - 1:
                next_token = tokens[i + 1].text.lower()
                if next_token in ORDER_OF_MAGNITUDE_WORDS:
                    magnitude = ORDER_OF_MAGNITUDE_WORDS[next_token]
                    token_text += ' ' + tokens[i + 1].text

            multiplier = 1
            if i < len(tokens) - 1:
                next_token = tokens[i + 1].text.lower()
                if next_token in ["%", "percent", "cents", "cent", "pennies", "penny"]:
                    multiplier = 0.01
                    token_text += ' ' + tokens[i + 1].text

            is_range = False
            if len(text) > 1 and text[-1] == 's' and text[-2] == '0':
                is_range = True
                text = text[:-1]

            # We strip out any non-digit characters, to capture things like '7th', or '1ghz'.  The
            # way we're doing this could lead to false positives for something like '1e2', but
            # we'll take that risk.  It shouldn't be a big deal.
            text = ''.join(text[i] for i, char in enumerate(text) if char in NUMBER_CHARACTERS)

            try:
                # We'll use a check for float(text) to find numbers, because text.isdigit() doesn't
                # catch things like "-3" or "0.07".
                number = float(text)
            except ValueError:
                pass

            if number is not None:
                #print("Number: ", number, " , multiplier: ", multiplier)
                number = number * magnitude * multiplier
                if float(number).is_integer():
                    number_string = '%d' % number
                else:
                    number_string = str(float(number))
                #print("Adding ", number_string)
                numbers.append((number_string, token_text))
                if is_range:
                    # TODO(mattg): both numbers in the range will have the same text, and so the
                    # linking score won't have any way to differentiate them...  We should figure
                    # out a better way to handle this.
                    num_zeros = 1
                    while text[-(num_zeros + 1)] == '0':
                        num_zeros += 1
                    numbers.append((str(int(number + 10 ** num_zeros)), token_text))

        # WARNING: To avoid empty entity sets, we add 2 dummy ones if less than 2 distinct numbers were extracted.
        keys = [x[0] for x in numbers]
        if len(set(keys)) < 1:
            numbers.append(("1", tokens[0].text))

        keys = [x[0] for x in numbers]
        if len(set(keys)) < 2:
            numbers.append(("1", tokens[0].text))
            numbers.append(("2", tokens[0].text))

        return numbers
