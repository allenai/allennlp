# allennlp.nn.chu_liu_edmonds

## decode_mst
```python
decode_mst(energy:numpy.ndarray, length:int, has_labels:bool=True) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Note: Counter to typical intuition, this function decodes the _maximum_
spanning tree.

Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
maximum spanning arborescences on graphs.

Parameters
----------
energy : ``numpy.ndarray``, required.
    A tensor with shape (num_labels, timesteps, timesteps)
    containing the energy of each edge. If has_labels is ``False``,
    the tensor should have shape (timesteps, timesteps) instead.
length : ``int``, required.
    The length of this sequence, as the energy may have come
    from a padded batch.
has_labels : ``bool``, optional, (default = True)
    Whether the graph has labels or not.

## chu_liu_edmonds
```python
chu_liu_edmonds(length:int, score_matrix:numpy.ndarray, current_nodes:List[bool], final_edges:Dict[int, int], old_input:numpy.ndarray, old_output:numpy.ndarray, representatives:List[Set[int]])
```

Applies the chu-liu-edmonds algorithm recursively
to a graph with edge weights defined by score_matrix.

Note that this function operates in place, so variables
will be modified.

Parameters
----------
length : ``int``, required.
    The number of nodes.
score_matrix : ``numpy.ndarray``, required.
    The score matrix representing the scores for pairs
    of nodes.
current_nodes : ``List[bool]``, required.
    The nodes which are representatives in the graph.
    A representative at it's most basic represents a node,
    but as the algorithm progresses, individual nodes will
    represent collapsed cycles in the graph.
final_edges : ``Dict[int, int]``, required.
    An empty dictionary which will be populated with the
    nodes which are connected in the maximum spanning tree.
old_input : ``numpy.ndarray``, required.
old_output : ``numpy.ndarray``, required.
representatives : ``List[Set[int]]``, required.
    A list containing the nodes that a particular node
    is representing at this iteration in the graph.

Returns
-------
Nothing - all variables are modified in place.


