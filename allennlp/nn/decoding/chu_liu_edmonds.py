
from typing import List, Set, Tuple, Dict
import numpy

from allennlp.common.checks import ConfigurationError

def decode_mst(energy: numpy.ndarray,
               length: int,
               leading_symbolic: int = 0,
               has_labels: bool = True):
    """
    Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
    minimum spanning arboresences on graphs.

    Parameters
    ----------
    energy : ``numpy.ndarray``, required.
        A tensor with shape (num_labels, timesteps, timesteps)
        containing the energy of each edge. If has_labels is ``False``,
        the tensor should have shape (timesteps, timesteps) instead.
    length : ``int``, required.
        The length of this sequence, as the energy may have come
        from a padded batch.
    leading_symbolic : ``int``, optional, (default = 0)
        The number of symbolic dependency types in the label space.
    has_labels : ``bool``, optional, (default = True)
        Whether the graph has labels or not.
    """
    if has_labels and energy.ndim != 3:
        raise ConfigurationError("The dimension of the energy array is not equal to 3.")
    elif not has_labels and energy.ndim != 2:
        raise ConfigurationError("The dimension of the energy array is not equal to 2.")
    input_shape = energy.shape
    max_length = input_shape[2]

    # Our energy matrix might have been batched -
    # here we clip it to contain only non padded tokens.
    if has_labels:
        energy = energy[leading_symbolic:, :length, :length]
        # get best label for each edge.
        label_id_matrix = energy.argmax(axis=0) + leading_symbolic
        energy = energy.max(axis=0)
    else:
        energy = energy[:length, :length]
        label_id_matrix = None
    # get original score matrix
    original_score_matrix = energy
    # initialize score matrix to original score matrix
    score_matrix = numpy.array(original_score_matrix, copy=True)

    old_input = numpy.zeros([length, length], dtype=numpy.int32)
    old_output = numpy.zeros([length, length], dtype=numpy.int32)
    current_nodes = numpy.zeros([length], dtype=numpy.bool)
    reps: List[Set[int]] = []

    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
        current_nodes[node1] = True
        reps.append({node1})

        for node2 in range(node1 + 1, length):
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2

            old_input[node2, node1] = node2
            old_output[node2, node1] = node1

    final_edges: Dict[int, int] = {}

    chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, reps)

    heads = numpy.zeros([max_length], numpy.int32)
    if has_labels:
        head_type = numpy.ones([max_length], numpy.int32)
        head_type[0] = 0
    else:
        head_type = None

    for child, parent in final_edges.items():
        heads[child] = parent
        if has_labels and child != 0:
            head_type[child] = label_id_matrix[parent, child]

    heads[0] = 0

    return heads, head_type

def _find_cycle(parents: numpy.ndarray,
                length: int,
                current_nodes: numpy.ndarray) -> Tuple[bool, Set[int]]:
    added = numpy.zeros([length], numpy.bool)
    added[0] = True
    cycle = set()
    has_cycle = False
    for i in range(1, length):
        if has_cycle:
            break
        # don't redo nodes we've already
        # visited or aren't considering.
        if added[i] or not current_nodes[i]:
            continue
        # Initialize a new possible cycle.
        this_cycle = set()
        this_cycle.add(i)
        added[i] = True
        has_cycle = True
        next_node = i

        while parents[next_node] not in this_cycle:
            next_node = parents[next_node]
            # If we see a node we've already processed,
            # we can stop, because the node we are
            # processing would have been in that cycle.
            if added[next_node]:
                has_cycle = False
                break
            added[next_node] = True
            this_cycle.add(next_node)

        if has_cycle:
            original = next_node
            cycle.add(original)
            next_node = parents[original]
            while next_node != original:
                cycle.add(next_node)
                next_node = parents[next_node]
            break

    return has_cycle, cycle

def chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, reps):
    parents = numpy.zeros([length], dtype=numpy.int32)
    # create best graph
    parents[0] = -1
    for node1 in range(1, length):
        if current_nodes[node1]:
            max_score = score_matrix[0, node1]
            parents[node1] = 0
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue

                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2

    # find a cycle
    has_cycle, cycle = _find_cycle(parents, length, current_nodes)
    # If there are no cycles, find all edges and return.
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue

            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        return

    cycle_length = len(cycle)
    cycle_weight = 0.0
    cycle_nodes = numpy.zeros([cycle_length], dtype=numpy.int32)
    index = 0
    for node in cycle:
        cycle_nodes[index] = node
        index += 1
        cycle_weight += score_matrix[parents[node], node]

    rep = cycle_nodes[0]
    for node in range(length):
        if not current_nodes[node] or node in cycle:
            continue

        max1 = float("-inf")
        wh1 = -1
        max2 = float("-inf")
        wh2 = -1

        for j in range(cycle_length):
            node_in_cycle = cycle_nodes[j]
            if score_matrix[node_in_cycle, node] > max1:
                max1 = score_matrix[node_in_cycle, node]
                wh1 = node_in_cycle

            score = cycle_weight + score_matrix[node, node_in_cycle] - score_matrix[parents[node_in_cycle], node_in_cycle]

            if score > max2:
                max2 = score
                wh2 = node_in_cycle

        score_matrix[rep, node] = max1
        old_input[rep, node] = old_input[wh1, node]
        old_output[rep, node] = old_output[wh1, node]
        score_matrix[node, rep] = max2
        old_output[node, rep] = old_output[node, wh2]
        old_input[node, rep] = old_input[node, wh2]

    rep_cons = []
    for i in range(cycle_length):
        rep_cons.append(set())

        cycle_node = cycle_nodes[i]
        for cc in reps[cycle_node]:
            rep_cons[i].add(cc)

    for i in range(1, cycle_length):
        cycle_node = cycle_nodes[i]
        current_nodes[cycle_node] = False
        for cc in reps[cycle_node]:
            reps[rep].add(cc)

    chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, reps)

    # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
    found = False
    wh = -1
    for i in range(cycle_length):
        for repc in rep_cons[i]:
            if repc in final_edges:
                wh = cycle_nodes[i]
                found = True
                break
        if found:
            break

    l = parents[wh]
    while l != wh:
        child = old_output[parents[l], l]
        parent = old_input[parents[l], l]
        final_edges[child] = parent
        l = parents[l]

