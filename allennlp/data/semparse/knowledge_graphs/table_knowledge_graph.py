"""
Classes related to representing a table in WikitableQuestions. At this point we have just a
``TableKnowledgeGraph``, a ``KnowledgeGraph`` that reads a TSV file and stores a table representation.
"""
import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from unidecode import unidecode

from allennlp.data.semparse.knowledge_graphs.knowledge_graph import KnowledgeGraph


class TableKnowledgeGraph(KnowledgeGraph):
    """
    Graph representation of the table. For now, we just store the neighborhood information of cells and
    columns. A column's neighbors are all the cells under it, and a cell's only neighbor is the column
    it is under. We store them all in a single dict. We don't have to worry about name clashes because we
    follow NLTK's naming convention for representing cells and columns, and thus they have unique names.

    This is a rather simplistic view of the table. For example, we don't store the order of rows.
    """
    # TODO (pradeep): We may want to reconsider this representation later.
    @classmethod
    def read_from_file(cls, filename: str) -> 'TableKnowledgeGraph':
        """
        We read tables formatted as TSV files here. We assume the first line in the file is a tab separated
        list of column headers, and all subsequent lines are content rows. For example if the TSV file is:

        Nation      Olympics    Medals
        USA         1896        8
        China       1932        9

        we read "Nation", "Olympics" and "Medals" as column headers, "USA" and "China" as cells under the
        "Nation" column and so on.
        """
        cells = []
        # We assume the first row is column names.
        for row_index, line in enumerate(open(filename)):
            line = line.rstrip('\n')
            if row_index == 0:
                columns = line.split('\t')
            else:
                cells.append(line.split('\t'))
        return cls.read_from_json({"columns": columns, "cells": cells})

    @classmethod
    def read_from_json(cls, json_object: Dict[str, Any]) -> 'TableKnowledgeGraph':
        """
        We read tables formatted as JSON objects (dicts) here. This is useful when you are reading data
        from a demo. The expected format is:

        {"columns": [column1, column2, ...],
         "cells": [[row1_cell1, row1_cell2, ...],
                   [row2_cell1, row2_cell2, ...],
                   ... ]}
        """
        entity_text: Dict[str, str] = {}
        neighbors: DefaultDict[str, List[str]] = defaultdict(list)
        # Following Sempre's convention for naming columns.  Sempre gives columns unique names when
        # columns normalize to a collision, so we keep track of these.  We do not give cell text
        # unique names, however, as `fb:cell.x` is actually a function that returns all cells that
        # have text that normalizes to "x".
        column_ids = []
        columns: Dict[str, int] = {}
        for column_string in json_object['columns']:
            normalized_string = f'fb:row.row.{cls._normalize_string(column_string)}'
            if normalized_string in columns:
                columns[normalized_string] += 1
                normalized_string = f'{normalized_string}_{columns[normalized_string]}'
            columns[normalized_string] = 1
            column_ids.append(normalized_string)
            entity_text[normalized_string] = column_string

        for row_index, row_cells in enumerate(json_object['cells']):
            assert len(columns) == len(row_cells), ("Invalid format. Row %d has %d cells, but header has %d"
                                                    " columns" % (row_index, len(row_cells), len(columns)))
            # Following Sempre's convention for naming cells.
            row_cell_ids = []
            for cell_string in row_cells:
                normalized_string = f'fb:cell.{cls._normalize_string(cell_string)}'
                row_cell_ids.append(normalized_string)
                entity_text[normalized_string] = cell_string
            for column, cell in zip(column_ids, row_cell_ids):
                neighbors[column].append(cell)
                neighbors[cell].append(column)
        return cls(set(neighbors.keys()), dict(neighbors), entity_text)

    @staticmethod
    def _normalize_string(string: str) -> str:
        """
        These are the transformation rules used to normalize cell in column names in Sempre.
        See ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
        ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.
        We reproduce those rules here to normalize and canonicalize cells and columns in the same way
        so that we can match them against constants in logical forms appropriately.
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
        string = re.sub("[ðø′″€⁄ª]", "_", string)
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
