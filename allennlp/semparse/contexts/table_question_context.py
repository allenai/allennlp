import re
import csv
from typing import Union, Dict, List, Tuple, Set
from collections import defaultdict

from unidecode import unidecode
from allennlp.data.tokenizers import Token
from allennlp.semparse.domain_languages.domain_language import ExecutionError
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph

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


class Date:
    def __init__(self, year: int, month: int, day: int) -> None:
        self.year = year
        self.month = month
        self.day = day

    def __eq__(self, other) -> bool:
        # Note that the logic below renders equality to be non-transitive. That is,
        # Date(2018, -1, -1) == Date(2018, 2, 3) and Date(2018, -1, -1) == Date(2018, 4, 5)
        # but Date(2018, 2, 3) != Date(2018, 4, 5).
        if not isinstance(other, Date):
            raise ExecutionError("only compare Dates with Dates")
        year_is_same = self.year == -1 or other.year == -1 or self.year == other.year
        month_is_same = self.month == -1 or other.month == -1 or self.month == other.month
        day_is_same = self.day == -1 or other.day == -1 or self.day == other.day
        return year_is_same and month_is_same and day_is_same

    def __gt__(self, other) -> bool:
        # pylint: disable=too-many-return-statements
        # The logic below is tricky, and is based on some assumptions we make about date comparison.
        # Year, month or day being -1 means that we do not know its value. In those cases, the
        # we consider the comparison to be undefined, and return False if all the fields that are
        # more significant than the field being compared are equal. However, when year is -1 for both
        # dates being compared, it is safe to assume that the year is not specified because it is
        # the same. So we make an exception just in that case. That is, we deem the comparison
        # undefined only when one of the year values is -1, but not both.
        if not isinstance(other, Date):
            raise ExecutionError("only compare Dates with Dates")
        # We're doing an exclusive or below.
        if (self.year == -1) != (other.year == -1):
            return False  # comparison undefined
        # If both years are -1, we proceed.
        if self.year != other.year:
            return self.year > other.year
        # The years are equal and not -1, or both are -1.
        if self.month == -1 or other.month == -1:
            return False
        if self.month != other.month:
            return self.month > other.month
        # The months and years are equal and not -1
        if self.day == -1 or other.day == -1:
            return False
        return self.day > other.day

    def __ge__(self, other) -> bool:
        if not isinstance(other, Date):
            raise ExecutionError("only compare Dates with Dates")
        return self > other or self == other

    def __str__(self):
        if (self.month, self.day) == (-1, -1):
            # If we have only the year, return just that so that the official evaluator does the
            # comparison against the target as if both are numbers.
            return str(self.year)
        return f"{self.year}-{self.month}-{self.day}"

    def __hash__(self):
        return hash(str(self))

    @classmethod
    def make_date(cls, string: str) -> 'Date':
        year_string, month_string, day_string = string.split("-")
        year = -1
        month = -1
        day = -1
        try:
            year = int(year_string)
        except ValueError:
            pass
        try:
            month = int(month_string)
        except ValueError:
            pass
        try:
            day = int(day_string)
        except ValueError:
            pass
        return Date(year, month, day)

CellValueType = Union[str, float, Date]  # pylint: disable=invalid-name


class TableQuestionContext:
    """
    A barebones implementation similar to
    https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/table/wtq/preprocess.py
    for extracting entities from a question given a table and type its columns with <string> | <date> | <number>
    """
    def __init__(self,
                 table_data: List[Dict[str, CellValueType]],
                 column_types: Dict[str, Set[str]],
                 column_names: Set[str],
                 question_tokens: List[Token]) -> None:
        self.table_data = table_data
        self.column_types: Set[str] = set()
        self.column_names = column_names
        for types in column_types.values():
            self.column_types.update(types)
        self.question_tokens = question_tokens
        # Mapping from strings to the columns they are under.
        string_column_mapping: Dict[str, List[str]] = defaultdict(list)
        for table_row in table_data:
            for column_name, cell_value in table_row.items():
                if "string_column:" in column_name and cell_value is not None:
                    string_column_mapping[cell_value].append(column_name)
        # We want the object to raise KeyError when checking if a specific string is a cell in the
        # table.
        self._string_column_mapping = dict(string_column_mapping)
        self._table_knowledge_graph: KnowledgeGraph = None

    def __eq__(self, other):
        if not isinstance(other, TableQuestionContext):
            return False
        return self.table_data == other.table_data

    def get_table_knowledge_graph(self) -> KnowledgeGraph:
        if self._table_knowledge_graph is None:
            entities: Set[str] = set()
            neighbors: Dict[str, List[str]] = defaultdict(list)
            entity_text: Dict[str, str] = {}
            # Add all column names to entities. We'll define their neighbors to be empty lists for
            # now, and later add number and string entities as needed.
            number_columns = []
            date_columns = []
            for typed_column_name in self.table_data[0].keys():
                if "number_column:" in typed_column_name or "num2_column" in typed_column_name:
                    number_columns.append(typed_column_name)

                if "date_column:" in typed_column_name:
                    date_columns.append(typed_column_name)

                # Add column names to entities, with no neighbors yet.
                entities.add(typed_column_name)
                neighbors[typed_column_name] = []
                entity_text[typed_column_name] = typed_column_name.split(":")[-1].replace("_", " ")

            string_entities, numbers = self.get_entities_from_question()
            for entity, column_names in string_entities:
                entities.add(entity)
                for column_name in column_names:
                    neighbors[entity].append(column_name)
                    neighbors[column_name].append(entity)
                entity_text[entity] = entity.replace("string:", "").replace("_", " ")
            # For all numbers (except -1), we add all number and date columns as their neighbors.
            for number, _ in numbers:
                entities.add(number)
                neighbors[number].extend(number_columns + date_columns)
                for column_name in number_columns + date_columns:
                    neighbors[column_name].append(number)
                entity_text[number] = number
            for entity, entity_neighbors in neighbors.items():
                neighbors[entity] = list(set(entity_neighbors))

            # Add "-1" as an entity only if we have date columns in the table because we will need
            # it as a wild-card in dates. The neighbors are the date columns.
            if "-1" not in neighbors and date_columns:
                entities.add("-1")
                neighbors["-1"] = date_columns
                entity_text["-1"] = "-1"
                for date_column in date_columns:
                    neighbors[date_column].append("-1")
            self._table_knowledge_graph = KnowledgeGraph(entities, dict(neighbors), entity_text)
        return self._table_knowledge_graph

    @classmethod
    def read_from_lines(cls,
                        lines: List[List[str]],
                        question_tokens: List[Token]) -> 'TableQuestionContext':
        column_index_to_name = {}

        header = lines[0] # the first line is the header
        index = 1
        # Each row is a mapping from column names to cell data. Cell data is a dict, where keys are
        # "string", "number", "num2" and "date", and the values are the corresponding values
        # extracted by CoreNLP.
        table_data: List[Dict[str, Dict[str, str]]] = []
        while lines[index][0] == '-1':
            # column names start with fb:row.row.
            current_line = lines[index]
            column_name_sempre = current_line[2]
            column_index = int(current_line[1])
            column_name = column_name_sempre.replace('fb:row.row.', '')
            column_index_to_name[column_index] = column_name
            index += 1
        column_types: Dict[str, Set[str]] = defaultdict(set)
        last_row_index = -1
        for current_line in lines[1:]:
            row_index = int(current_line[0])
            if row_index == -1:
                continue  # header row
            column_index = int(current_line[1])
            if row_index != last_row_index:
                table_data.append({})
            node_info = dict(zip(header, current_line))
            cell_data: Dict[str, str] = {}
            column_name = column_index_to_name[column_index]
            if node_info['date']:
                column_types[column_name].add("date")
                cell_data["date"] = node_info["date"]

            if node_info['number']:
                column_types[column_name].add("number")
                cell_data["number"] = node_info["number"]

            if node_info['num2']:
                column_types[column_name].add("num2")
                cell_data["num2"] = node_info["num2"]

            if node_info['content'] != '—':
                column_types[column_name].add("string")
                cell_data['string'] = node_info["content"]

            table_data[-1][column_name] = cell_data
            last_row_index = row_index
        # Table data with each column split into different ones, depending on the types they have.
        table_data_with_column_types: List[Dict[str, CellValueType]] = []
        all_column_names: Set[str] = set()
        for table_row in table_data:
            table_data_with_column_types.append({})
            for column_name, cell_data in table_row.items():
                for column_type in column_types[column_name]:
                    typed_column_name = f"{column_type}_column:{column_name}"
                    all_column_names.add(typed_column_name)
                    cell_value_string = cell_data.get(column_type, None)
                    if column_type in ["number", "num2"]:
                        try:
                            cell_number = float(cell_value_string)
                        except (ValueError, TypeError):
                            cell_number = None
                        table_data_with_column_types[-1][typed_column_name] = cell_number
                    elif column_type == "date":
                        cell_date = None
                        if cell_value_string is not None:
                            cell_date = Date.make_date(cell_value_string)
                        table_data_with_column_types[-1][typed_column_name] = cell_date
                    else:
                        if cell_value_string is None:
                            normalized_string = None
                        else:
                            normalized_string = cls.normalize_string(cell_value_string)
                        table_data_with_column_types[-1][typed_column_name] = normalized_string
        return cls(table_data_with_column_types, column_types, all_column_names, question_tokens)

    @classmethod
    def read_from_file(cls, filename: str, question_tokens: List[Token]) -> 'TableQuestionContext':
        with open(filename, 'r') as file_pointer:
            reader = csv.reader(file_pointer, delimiter='\t', quoting=csv.QUOTE_NONE)
            lines = [line for line in reader]
            return cls.read_from_lines(lines, question_tokens)

    def get_entities_from_question(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, int]]]:
        entity_data = []
        for i, token in enumerate(self.question_tokens):
            token_text = token.text
            if token_text in STOP_WORDS:
                continue
            normalized_token_text = self.normalize_string(token_text)
            if not normalized_token_text:
                continue
            token_columns = self._string_in_table(normalized_token_text)
            if token_columns:
                token_type = token_columns[0].split(":")[0].replace("_column", "")
                entity_data.append({'value': normalized_token_text,
                                    'token_start': i,
                                    'token_end': i+1,
                                    'token_type': token_type,
                                    'token_in_columns': token_columns})

        extracted_numbers = self._get_numbers_from_tokens(self.question_tokens)
        # filter out number entities to avoid repetition
        expanded_entities = []
        for entity in self._expand_entities(self.question_tokens, entity_data):
            if entity["token_type"] == "string":
                expanded_entities.append((f"string:{entity['value']}", entity['token_in_columns']))
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
            token_text = token.text
            text = token.text.replace(',', '').lower()
            number = NUMBER_WORDS.get(text, None)

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

    def _string_in_table(self, candidate: str) -> List[str]:
        """
        Checks if the string occurs in the table, and if it does, returns the names of the columns
        under which it occurs. If it does not, returns an empty list.
        """
        candidate_column_names: List[str] = []
        # First check if the entire candidate occurs as a cell.
        if candidate in self._string_column_mapping:
            candidate_column_names = self._string_column_mapping[candidate]
        # If not, check if it is a substring pf any cell value.
        if not candidate_column_names:
            for cell_value, column_names in self._string_column_mapping.items():
                if candidate in cell_value:
                    candidate_column_names.extend(column_names)
        candidate_column_names = list(set(candidate_column_names))
        return candidate_column_names

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
            current_token_columns = entity['token_in_columns']

            while current_end < len(question):
                next_token = question[current_end].text
                next_token_normalized = self.normalize_string(next_token)
                if next_token_normalized == "":
                    current_end += 1
                    continue
                candidate = "%s_%s" %(current_token, next_token_normalized)
                candidate_columns = self._string_in_table(candidate)
                candidate_columns = list(set(candidate_columns).intersection(current_token_columns))
                if not candidate_columns:
                    break
                candidate_type = candidate_columns[0].split(":")[0].replace("_column", "")
                if candidate_type != current_token_type:
                    break
                current_end += 1
                current_token = candidate
                current_token_columns = candidate_columns

            new_entities.append({'token_start' : current_start,
                                 'token_end' : current_end,
                                 'value' : current_token,
                                 'token_type': current_token_type,
                                 'token_in_columns': current_token_columns})
        return new_entities

    @staticmethod
    def normalize_string(string: str) -> str:
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
        # Canonicalization rules from Sempre.
        string = re.sub("[^\\w]", "_", string)
        string = re.sub("_+", "_", string)
        string = re.sub("_$", "", string)
        return unidecode(string.lower())
