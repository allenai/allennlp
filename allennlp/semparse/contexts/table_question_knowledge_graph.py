import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple, Union, Set

from overrides import overrides
from unidecode import unidecode

from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph

DEFAULT_NUMBERS = ['-1', '0', '1']
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


class TableQuestionKnowledgeGraph(KnowledgeGraph):
    """
    A ``TableQuestionKnowledgeGraph`` represents the linkable entities in a table and a question
    about the table.  The linkable entities in a table are the cells and the columns of the table,
    and the linkable entities from the question are the numbers in the question.  We use the
    question to define our space of allowable numbers, because there are infinitely many numbers
    that we could include in our action space, and we really don't want to do that. Additionally, we
    have a method that returns the set of entities in the graph that are relevant to the question,
    and we keep the question for this method. See ``get_linked_agenda_items`` for more information.

    To represent the table as a graph, we make each cell and column a node in the graph, and
    consider a column's neighbors to be all cells in that column (and thus each cell has just one
    neighbor - the column it belongs to).  This is a rather simplistic view of the table. For
    example, we don't store the order of rows.

    We represent numbers as standalone nodes in the graph, without any neighbors.

    Additionally, when we encounter cells that can be split, we create ``fb:part.[something]``
    entities, also without any neighbors.
    """
    def __init__(self,
                 entities: Set[str],
                 neighbors: Dict[str, List[str]],
                 entity_text: Dict[str, str],
                 question_tokens: List[Token]) -> None:
        super().__init__(entities, neighbors, entity_text)
        self.question_tokens = question_tokens
        self._entity_prefixes: Dict[str, List[str]] = defaultdict(list)
        for entity, text in self.entity_text.items():
            parts = text.split()
            if not parts:
                continue
            prefix = parts[0].lower()
            self._entity_prefixes[prefix].append(entity)

    @classmethod
    def read_from_file(cls, filename: str, question: List[Token]) -> 'TableQuestionKnowledgeGraph':
        """
        We read tables formatted as TSV files here. We assume the first line in the file is a tab
        separated list of column headers, and all subsequent lines are content rows. For example if
        the TSV file is:

        Nation      Olympics    Medals
        USA         1896        8
        China       1932        9

        we read "Nation", "Olympics" and "Medals" as column headers, "USA" and "China" as cells
        under the "Nation" column and so on.
        """
        return cls.read_from_lines(open(filename).readlines(), question)

    @classmethod
    def read_from_lines(cls, lines: List[str], question: List[Token]) -> 'TableQuestionKnowledgeGraph':
        cells = []
        # We assume the first row is column names.
        for row_index, line in enumerate(lines):
            line = line.rstrip('\n')
            if row_index == 0:
                columns = line.split('\t')
            else:
                cells.append(line.split('\t'))
        return cls.read_from_json({"columns": columns, "cells": cells, "question": question})

    @classmethod
    def read_from_json(cls, json_object: Dict[str, Any]) -> 'TableQuestionKnowledgeGraph':
        """
        We read tables formatted as JSON objects (dicts) here. This is useful when you are reading
        data from a demo. The expected format is::

            {"question": [token1, token2, ...],
             "columns": [column1, column2, ...],
             "cells": [[row1_cell1, row1_cell2, ...],
                       [row2_cell1, row2_cell2, ...],
                       ... ]}
        """
        entity_text: Dict[str, str] = {}
        neighbors: DefaultDict[str, List[str]] = defaultdict(list)

        # Getting number entities first.  Number entities don't have any neighbors, and their
        # "entity text" is the text from the question that evoked the number.
        question_tokens = json_object['question']
        for number, number_text in cls._get_numbers_from_tokens(question_tokens):
            entity_text[number] = number_text
            neighbors[number] = []
        for default_number in DEFAULT_NUMBERS:
            if default_number not in neighbors:
                neighbors[default_number] = []
                entity_text[default_number] = default_number

        # Following Sempre's convention for naming columns.  Sempre gives columns unique names when
        # columns normalize to a collision, so we keep track of these.  We do not give cell text
        # unique names, however, as `fb:cell.x` is actually a function that returns all cells that
        # have text that normalizes to "x".
        column_ids = []
        columns: Dict[str, int] = {}
        for column_string in json_object['columns']:
            column_string = column_string.replace('\\n', '\n')
            normalized_string = f'fb:row.row.{cls._normalize_string(column_string)}'
            if normalized_string in columns:
                columns[normalized_string] += 1
                normalized_string = f'{normalized_string}_{columns[normalized_string]}'
            columns[normalized_string] = 1
            column_ids.append(normalized_string)
            entity_text[normalized_string] = column_string

        # Stores cell text to cell name, making sure that unique text maps to a unique name.
        cell_id_mapping: Dict[str, str] = {}
        column_cells: List[List[str]] = [[] for _ in columns]
        for row_index, row_cells in enumerate(json_object['cells']):
            assert len(columns) == len(row_cells), ("Invalid format. Row %d has %d cells, but header has %d"
                                                    " columns" % (row_index, len(row_cells), len(columns)))
            # Following Sempre's convention for naming cells.
            row_cell_ids = []
            for column_index, cell_string in enumerate(row_cells):
                cell_string = cell_string.replace('\\n', '\n')
                column_cells[column_index].append(cell_string)
                if cell_string in cell_id_mapping:
                    normalized_string = cell_id_mapping[cell_string]
                else:
                    base_normalized_string = f'fb:cell.{cls._normalize_string(cell_string)}'
                    normalized_string = base_normalized_string
                    attempt_number = 1
                    while normalized_string in cell_id_mapping.values():
                        attempt_number += 1
                        normalized_string = f"{base_normalized_string}_{attempt_number}"
                    cell_id_mapping[cell_string] = normalized_string
                row_cell_ids.append(normalized_string)
                entity_text[normalized_string] = cell_string
            for column_id, cell_id in zip(column_ids, row_cell_ids):
                neighbors[column_id].append(cell_id)
                neighbors[cell_id].append(column_id)

        for column in column_cells:
            if cls._should_split_column_cells(column):
                for cell_string in column:
                    for part_entity, part_string in cls._get_cell_parts(cell_string):
                        neighbors[part_entity] = []
                        entity_text[part_entity] = part_string
        return cls(set(neighbors.keys()), dict(neighbors), entity_text, question_tokens)

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
        # Canonicalization rules from Sempre
        string = re.sub("[^\\w]", "_", string)
        string = re.sub("_+", "_", string)
        string = re.sub("_$", "", string)
        return unidecode(string.lower())

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
                numbers.append((number_string, token_text))
                if is_range:
                    # TODO(mattg): both numbers in the range will have the same text, and so the
                    # linking score won't have any way to differentiate them...  We should figure
                    # out a better way to handle this.
                    num_zeros = 1
                    while text[-(num_zeros + 1)] == '0':
                        num_zeros += 1
                    numbers.append((str(int(number + 10 ** num_zeros)), token_text))
        return numbers

    cell_part_regex = re.compile(r',\s|\n|/')
    @classmethod
    def _get_cell_parts(cls, cell_text: str) -> List[Tuple[str, str]]:
        """
        Splits a cell into parts and returns the parts of the cell.  We return a list of
        ``(entity_name, entity_text)``, where ``entity_name`` is ``fb:part.[something]``, and
        ``entity_text`` is the text of the cell corresponding to that part.  For many cells, there
        is only one "part", and we return a list of length one.

        Note that you shouldn't call this on every cell in the table; SEMPRE decides to make these
        splits only when at least one of the cells in a column looks "splittable".  Only if you're
        splitting the cells in a column should you use this function.
        """
        parts = []
        for part_text in cls.cell_part_regex.split(cell_text):
            part_text = part_text.strip()
            part_entity = f'fb:part.{cls._normalize_string(part_text)}'
            parts.append((part_entity, part_text))
        return parts

    @classmethod
    def _should_split_column_cells(cls, column_cells: List[str]) -> bool:
        """
        Returns true if there is any cell in this column that can be split.
        """
        return any(cls._should_split_cell(cell_text) for cell_text in column_cells)

    @classmethod
    def _should_split_cell(cls, cell_text: str) -> bool:
        """
        Checks whether the cell should be split.  We're just doing the same thing that SEMPRE did
        here.
        """
        if ', ' in cell_text or '\n' in cell_text or '/' in cell_text:
            return True
        return False

    def get_linked_agenda_items(self) -> List[str]:
        """
        Returns entities that can be linked to spans in the question, that should be in the agenda,
        for training a coverage based semantic parser. This method essentially does a heuristic
        entity linking, to provide weak supervision for a learning to search parser.
        """
        agenda_items: List[str] = []
        for entity in self._get_longest_span_matching_entities():
            agenda_items.append(entity)
            # If the entity is a cell, we need to add the column to the agenda as well,
            # because the answer most likely involves getting the row with the cell.
            if 'fb:cell' in entity:
                agenda_items.append(self.neighbors[entity][0])
        return agenda_items

    def _get_longest_span_matching_entities(self):
        question = " ".join([token.text for token in self.question_tokens])
        matches_starting_at: Dict[int, List[str]] = defaultdict(list)
        for index, token in enumerate(self.question_tokens):
            if token.text in self._entity_prefixes:
                for entity in self._entity_prefixes[token.text]:
                    if self.entity_text[entity].lower() in question:
                        matches_starting_at[index].append(entity)
        longest_matches: List[str] = []
        for index, matches in matches_starting_at.items():
            longest_matches.append(sorted(matches, key=len)[-1])
        return longest_matches

    @overrides
    def __eq__(self, other):
        if isinstance(self, other.__class__):
            for key in self.__dict__:
                # We need to specially handle question tokens because they are Spacy's ``Token``
                # objects, and equality is not defined for them.
                if key == "question_tokens":
                    self_tokens = self.__dict__[key]
                    other_tokens = other.__dict__[key]
                    if not all([token1.text == token2.text
                                for token1, token2 in zip(self_tokens, other_tokens)]):
                        return False
                else:
                    if not self.__dict__[key] == other.__dict__[key]:
                        return False
            return True
        return NotImplemented
