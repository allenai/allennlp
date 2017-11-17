"""
A ``KnowledgeGraph`` is a graphical representation of some structured knowledge source: say a table, figure
or an explicit knowledge base.
"""

from typing import Dict, List


class KnowledgeGraph:
    """
    ``KnowledgeGraph`` currently stores some basic neighborhood information, and provides  minimal
    functionality for querying that information, for embedding this knowledge or linking the entities in the
    knowledge base to some text.
    The knowledge base itself can be a table (like in WikitableQuestions), a figure (like in NLVR) or some
    other structured knowledge source. This abstract class needs to be inherited for implementing the
    functionality appropriate for a given KB.

    Parameters
    ----------
    neighbors : Dict[str, List[str]]
        Entities (represented as strings) mapped to their neighbors.
    """
    def __init__(self, neighbors: Dict[str, List[str]]) -> None:
        self._neighbors = neighbors

    @classmethod
    def read_from_file(cls, filename: str):
        raise NotImplementedError

    def get_neighbors(self, entity: str) -> List[str]:
        """
        Parameters
        ----------
        entity : str
            String representation of the entity whose neighbors will be returned.
        """
        return self._neighbors[entity]

    def get_all_entities(self):
        return self._neighbors.keys()
