import re

from collections import defaultdict
from typing import List, Dict


class TableKnowledgeGraph:
    """
    Graph representation of the table. For now, we just store the neighborhood information of cells and
    columns. A column's neighbors are all the cells under it, and a cell's only neighbor is the column
    it is under. This is a rather simplistic view of the table. For example, we don't store the order
    of rows, and we do not distinguish between multiple occurrences of the same cell name (we treat all
    those cells as the same entity). We may want to reconsider this later.

    Parameters
    ----------
    column_neighbors : Dict[str, List[str]]
        All the cells related to each column. Keys are column names and values are lists of cell names.
    cell_neighbors: Dict[str, List[str]]
        All the columns under which each cell name occurs. Keys are cell names and values are lists of
        column names.
    """
    def __init__(self,
                 column_neighbors: Dict[str, List[str]],
                 cell_neighbors: Dict[str, List[str]]) -> None:
        self._column_neighbors = column_neighbors
        self._cell_neighbors = cell_neighbors

    @classmethod
    def read_table_from_tsv(cls, table_filename: str) -> 'TableKnowledgeGraph':
        """
        We read tables formatted as TSV files here. We assume the first line in the file is a tab separated
        list of column headers, and all subsequent lines are content rows. For example if the TSV file is,
            Nation      Olympics    Medals
            USA         1896        8
            China       1932        9

        we read "Nation", "Olympics" and "Medals" as column headers, "USA" and "China" as cells under the
        "Nation" column and so on.
        """
        _column_neighbors = defaultdict(list)
        _cell_neighbors = defaultdict(list)
        columns = []
        # We assume the first row is column names.
        for row_index, line in enumerate(open(table_filename)):
            line = line.rstrip('\n')
            if row_index == 0:
                # Following Sempre's convention for naming columns.
                columns = ["fb:row.row.%s" % cls._normalize_string(x) for x in line.split('\t')]
            else:
                # Following Sempre's convention for naming cells.
                cells = ["fb:cell.%s" % cls._normalize_string(x) for x in line.split('\t')]
                assert len(columns) == len(cells), ("Invalid format. Row %d has %d columns, but header "
                                                    "has %d columns" % (row_index, len(cells), len(columns)))
                for column, cell in zip(columns, cells):
                    _column_neighbors[column].append(cell)
                    _cell_neighbors[cell].append(column)
        return cls(dict(_column_neighbors), dict(_cell_neighbors))

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
        string = re.sub("[•†‡]", "", string)
        string = re.sub("[‐‑–—]", "-", string)
        string = re.sub("[\\u2E00-\\uFFFF]", "", string)
        string = re.sub("\\s+", " ", string)
        # Canonicalization rules from Sempre
        string = re.sub("[^\\w]", "_", string)
        string = re.sub("_+", "_", string)
        string = re.sub("_$", "", string)
        return string.lower()

    def get_cell_neighbors(self, cell: str) -> List[str]:
        """
        Parameters
        ----------
        cell : str
            Sempre name of the cell (Eg. fb:cell.usa)
        """
        return self._cell_neighbors[cell]

    def get_column_neighbors(self, column: str) -> List[str]:
        """
        Parameters
        ----------
        column : str
            Sempre name of the column (Eg. fb:row.row.nation)
        """
        return self._column_neighbors[column]
