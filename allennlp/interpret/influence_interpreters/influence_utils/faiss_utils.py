import faiss
import numpy as np
from typing import Optional, Tuple


class FAISSIndex(object):
    def __init__(self,
                 d: Optional[int] = None,
                 description: Optional[str] = "Flat",
                 index: Optional[faiss.Index] = None) -> None:
        if index is None:
            index = faiss.index_factory(d, description)
        self._index = index

    def search(self,
               k: int,
               key: Optional[int] = None,
               query: Optional[np.ndarray] = None,
               queries: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search Nearest Neighbor

        Args:
            k [int]: number of nearest neighbors to use
            key [int, Optional]: Exclusive with `query` and `queries`,
                the key used to do the search if exists
                in the `index` already.
            query [int, Array <d>]: Exclusive with `key` and `queries`,
                the query used to do the search.
            queries [int, Array <n, d>]: Exclusive with `key` and `query`,
                the queries used to do the search.

        Returns:
            distance [Array <n, k>]: Distance of the return indices.
            indices [Array <n, k>]: Return indices.

        """
        num_inputs_provided = sum([
            key is not None,
            query is not None,
            queries is not None
        ])
        if num_inputs_provided != 1:
            raise ValueError
        query_expanded = None
        # if `query` is not None then we are good
        if key is not None:
            query = self.get(key)
            query_expanded = np.expand_dims(query, axis=0)

        if query is not None:
            query_expanded = np.expand_dims(query, axis=0)

        if queries is not None:
            query_expanded = queries
        assert query_expanded is not None
        if query_expanded.ndim != 2:
            raise ValueError

        return self._index.search(query_expanded, k)

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(vectors)

    def get(self, key: int) -> np.ndarray:
        """Returns Array <d>"""
        return self._index.reconstruct(key=key)

    def get_n(self, key_0: int, key_i: int) -> np.ndarray:
        """Returns Array <n, d>"""
        return self._index.reconstruct_n(
            n0=key_0, ni=key_i)

    def save(self, file_name) -> None:
        faiss.write_index(self._index, file_name)

    def load(self, file_name) -> None:
        self._index = faiss.read_index(file_name)

    def __len__(self):
        return self._index.ntotal
