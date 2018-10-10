import re
import csv
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

from unidecode import unidecode
from allennlp.data.tokenizers import Token

# == stop words that will be omitted by ContextGenerator
STOP_WORDS = {"", "", "all", "being", "-", "over", "through", "yourselves", "its", "before",
              "hadn", "with", "had", ",", "should", "to", "only", "under", "ours", "has", "ought", "do",
              "them", "his", "than", "very", "cannot", "they", "not", "during", "yourself", "him",
              "nor", "did", "didn", "'ve", "this", "she", "each", "where", "because", "doing", "some", "we", "are",
              "further", "ourselves", "out", "what", "for", "weren", "does", "above", "between", "mustn", "?",
              "be", "hasn", "who", "were", "here", "shouldn", "let", "hers", "by", "both", "about", "couldn",
              "of", "could", "against", "isn", "or", "own", "into", "while", "whom", "down", "wasn", "your",
              "from", "her", "their", "aren", "there", "been", ".", "few", "too", "wouldn", "themselves",
              ":", "was", "until", "more", "himself", "on", "but", "don", "herself", "haven", "those", "he",
              "me", "myself", "these", "up", ";", "below", "'re", "can", "theirs", "my", "and", "would", "then",
              "is", "am", "it", "doesn", "an", "as", "itself", "at", "have", "in", "any", "if", "!",
              "again", "'ll", "no", "that", "when", "same", "how", "other", "which", "you", "many", "shan",
              "'t", "'s", "our", "after", "most", "'d", "such", "'m", "why", "a", "off", "i", "yours", "so",
              "the", "having", "once"}

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


class TableQuestionContext:
    """
    A barebones implementation similar to
    https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/table/wtq/preprocess.py
    for extracting entities from a question given a table and type its columns with <string> | <date> | <number>
    """
    def __init__(self,
                 table_data: List[Dict[str, str]],
                 column_types: Dict[str, str],
                 question_tokens: List[Token]) -> None:
        self.table_data = table_data
        self.column_types = column_types
        self.question_tokens = question_tokens
        # Mapping from cell values to the types of the columns they are under.
        cell_values_with_types: Dict[str, List[str]] = defaultdict(list)
        for table_row in table_data:
            for column_name, cell_value in table_row.items():
                # "string_column:name" -> "string"
                column_type = column_name.split(":")[0].replace("_column", "")
                cell_values_with_types[cell_value].append(column_type)
        # We want the object to raise KeyError when checking if a specific string is a cell in the
        # table.
        self._cell_values_with_types = dict(cell_values_with_types)

    MAX_TOKENS_FOR_NUM_CELL = 2

    @classmethod
    def read_from_lines(cls,
                        lines: List[List[str]],
                        question_tokens: List[Token]) -> 'TableQuestionContext':
        column_index_to_name = {}

        header = lines[0] # the first line is the header
        index = 1
        table_data: List[Dict[str, str]] = []
        while lines[index][0] == '-1':
            # column names start with fb:row.row.
            current_line = lines[index]
            column_name_sempre = current_line[2]
            column_index = int(current_line[1])
            column_name = column_name_sempre.replace('fb:row.row.', '')
            column_index_to_name[column_index] = column_name
            index += 1
        column_node_type_info = [{'string' : 0, 'number' : 0, 'date' : 0}
                                 for col in column_index_to_name]
        last_row_index = -1
        for current_line in lines[1:]:
            row_index = int(current_line[0])
            if row_index == -1:
                continue  # header row
            column_index = int(current_line[1])
            if row_index != last_row_index:
                table_data.append({})
            node_info = dict(zip(header, current_line))
            cell_value = cls._normalize_string(node_info['content'])
            column_name = column_index_to_name[column_index]
            table_data[-1][column_name] = cell_value
            num_tokens = len(node_info['tokens'].split('|'))
            if node_info['date']:
                column_node_type_info[column_index]['date'] += 1
            # If cell contains too many tokens, then likely not number
            elif node_info['number'] and num_tokens <= cls.MAX_TOKENS_FOR_NUM_CELL:
                column_node_type_info[column_index]['number'] += 1
            elif node_info['content'] != '—':
                column_node_type_info[column_index]['string'] += 1
            last_row_index = row_index
        column_types: Dict[str, str] = {}
        for column_index, column_name in column_index_to_name.items():
            current_column_type_info = column_node_type_info[column_index]
            if current_column_type_info["string"] > 0:
                # There is at least one value that is neither date nor number.
                column_types[column_name] = "string"
            elif current_column_type_info["date"] > current_column_type_info["number"]:
                column_types[column_name] = "date"
            else:
                column_types[column_name] = "number"
        # Now that we determined the column types, let us prefix the column names with those.
        table_data_with_column_types: List[Dict[str, str]] = []
        for table_row in table_data:
            table_data_with_column_types.append({})
            for column_name, cell_value in table_row.items():
                column_type = column_types[column_name]
                typed_column_name = f"{column_type}_column:{column_name}"
                table_data_with_column_types[-1][typed_column_name] = cell_value
        return cls(table_data_with_column_types, column_types, question_tokens)

    @classmethod
    def read_from_file(cls, filename: str, question_tokens: List[Token]) -> 'TableQuestionContext':
        with open(filename, 'r') as file_pointer:
            reader = csv.reader(file_pointer, delimiter='\t', quoting=csv.QUOTE_NONE)
            lines = [line for line in reader]
            return cls.read_from_lines(lines, question_tokens)

    def get_entities_from_question(self):
        entity_data = []
        for i, token in enumerate(self.question_tokens):
            token_text = token.text
            if token_text in STOP_WORDS:
                continue
            normalized_token_text = self._normalize_string(token_text)
            if not normalized_token_text:
                continue
            token_type_in_table = self._string_type_in_table(normalized_token_text)
            if token_type_in_table:
                entity_data.append({'value': normalized_token_text,
                                    'token_start': i,
                                    'token_end': i+1,
                                    'token_type': token_type_in_table})

        extracted_numbers = self._get_numbers_from_tokens(self.question_tokens)
        # filter out number entities to avoid repetition
        expanded_entities = []
        for entity in self._expand_entities(self.question_tokens, entity_data):
            if entity["token_type"] == "string":
                expanded_entities.append(entity["value"])
        return expanded_entities, extracted_numbers  #TODO(shikhar) Handle conjunctions


    @staticmethod
    def _get_numbers_from_tokens(tokens: List[Token]) -> List[Tuple[str, int]]:
        """
        Finds numbers in the input tokens and returns them as strings.  We do some simple heuristic
        number recognition, finding ordinals and cardinals expressed as text ("one", "first",
        etc.), as well as numerals ("7th", "3rd"), months (mapping "july" to 7), and units
        ("1ghz").

        We also handle year ranges expressed as decade or centuries ("1800s" or "1950s"), adding
        the endpoints of the range as possible numbers to generate.

        We return a list of tuples, where each tuple is the (number_string, token_index) for a
        number found in the input tokens.
        """
        numbers = []
        for i, token in enumerate(tokens):
            number: Union[int, float] = None
            token_text = token.text
            text = token.text.replace(',', '').lower()
            if text in NUMBER_WORDS:
                number = NUMBER_WORDS[text]

            magnitude = 1
            if i < len(tokens) - 1:
                next_token = tokens[i + 1].text.lower()
                if next_token in ORDER_OF_MAGNITUDE_WORDS:
                    magnitude = ORDER_OF_MAGNITUDE_WORDS[next_token]
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
                number = number * magnitude
                if '.' in text:
                    number_string = '%.3f' % number
                else:
                    number_string = '%d' % number
                numbers.append((number_string, i))
                if is_range:
                    # TODO(mattg): both numbers in the range will have the same text, and so the
                    # linking score won't have any way to differentiate them...  We should figure
                    # out a better way to handle this.
                    num_zeros = 1
                    while text[-(num_zeros + 1)] == '0':
                        num_zeros += 1
                    numbers.append((str(int(number + 10 ** num_zeros)), i))
        return numbers

    def _string_type_in_table(self, candidate: str) -> Optional[str]:
        """
        Checks if the string occurs in the table, and if it does, returns the type of the column
        under which it occurs. If it does not, returns None.
        """
        candidate_column_types: List[str] = []
        # First check if the entire candidate occurs as a cell.
        if candidate in self._cell_values_with_types:
            candidate_column_types = self._cell_values_with_types[candidate]
        # If not, check if it is a substring pf any cell value.
        if not candidate_column_types:
            for cell_value, column_types in self._cell_values_with_types.items():
                if candidate in cell_value:
                    candidate_column_types = column_types
                    break
        if not candidate_column_types:
            return None
        candidate_column_types = list(set(candidate_column_types))
        # Note: if candidate_column_types is now a list with more than one element, it means that
        # the candidate matched a cell value that occurs under columns with multiple types. This is
        # unusual, and we are ignoring such cases, and simply returning the first type.
        return candidate_column_types[0]

    def _process_conjunction(self, entity_data):
        raise NotImplementedError

    def _expand_entities(self, question, entity_data):
        new_entities = []
        for entity in entity_data:
            # to ensure the same strings are not used over and over
            if new_entities and entity['token_end'] <= new_entities[-1]['token_end']:
                continue
            current_start = entity['token_start']
            current_end = entity['token_end']
            current_token = entity['value']
            current_token_type = entity['token_type']

            while current_end < len(question):
                next_token = question[current_end].text
                next_token_normalized = self._normalize_string(next_token)
                if next_token_normalized == "":
                    current_end += 1
                    continue
                candidate = "%s_%s" %(current_token, next_token_normalized)
                candidate_type = self._string_type_in_table(candidate)
                if candidate_type is not None and candidate_type == current_token_type:
                    current_end += 1
                    current_token = candidate
                else:
                    break

            new_entities.append({'token_start' : current_start,
                                 'token_end' : current_end,
                                 'value' : current_token,
                                 'token_type': current_token_type})
        return new_entities

    @staticmethod
    def _normalize_string(string: str) -> str:
        """
        These are the transformation rules used to normalize cell in column names in Sempre.  See
        ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
        ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.  We reproduce those
        rules here to normalize and canonicalize cells and columns in the same way so that we can
        match them against constants in logical forms appropriately.
        """
        # Normalization rules from Sempre
        # \u201A -> ,
        string = re.sub("‚", ",", string)
        string = re.sub("„", ",,", string)
        string = re.sub("[·・]", ".", string)
        string = re.sub("…", "...", string)
        string = re.sub("ˆ", "^", string)
        string = re.sub("˜", "~", string)
        string = re.sub("‹", "<", string)
        string = re.sub("›", ">", string)
        string = re.sub("[‘’´`]", "'", string)
        string = re.sub("[“”«»]", "\"", string)
        string = re.sub("[•†‡²³]", "", string)
        string = re.sub("[‐‑–—−]", "-", string)
        # Oddly, some unicode characters get converted to _ instead of being stripped.  Not really
        # sure how sempre decides what to do with these...  TODO(mattg): can we just get rid of the
        # need for this function somehow?  It's causing a whole lot of headaches.
        string = re.sub("[ðø′″€⁄ªΣ]", "_", string)
        # This is such a mess.  There isn't just a block of unicode that we can strip out, because
        # sometimes sempre just strips diacritics...  We'll try stripping out a few separate
        # blocks, skipping the ones that sempre skips...
        string = re.sub("[\\u0180-\\u0210]", "", string).strip()
        string = re.sub("[\\u0220-\\uFFFF]", "", string).strip()
        string = string.replace("\\n", "_")
        string = re.sub("\\s+", " ", string)
        # Canonicalization rules from Sempre. We changed it to let dots be if there are numbers in
        # the string, because the dots could be decimal points.
        if re.match("[0-9]+", string):
            string = re.sub("[^\\w.]", "_", string)
        else:
            string = re.sub("[^\\w]", "_", string)
        string = re.sub("_+", "_", string)
        string = re.sub("_$", "", string)
        return unidecode(string.lower())
