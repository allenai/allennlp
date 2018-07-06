
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
    current_nodes = [True for _ in range(length)]
    reps: List[Set[int]] = []

    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
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

def _find_cycle(parents: List[int],
                length: int,
                current_nodes: numpy.ndarray) -> Tuple[bool, List[int]]:
    added = [False for _ in range(length)]
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

    return has_cycle, list(cycle)

def chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, reps):
    # Set the initial graph to be the greedy best one.
    parents = [-1]
    for node1 in range(1, length):
        parents.append(0)
        if current_nodes[node1]:
            max_score = score_matrix[0, node1]
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue

                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2

    # Check if this solution has a cycle.
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

    # Otherwise, we have a cycle so we need to remove an edge.
    # From here until the recursive call is the contraction stage of the algorithm.
    cycle_weight = 0.0
    # Find the weight of the cycle.
    index = 0
    for node in cycle:
        index += 1
        cycle_weight += score_matrix[parents[node], node]

    # For each node in the graph, find the maximum weight incoming
    # and outgoing edge into the cycle.
    cycle_representative = cycle[0]
    for node in range(length):
        if not current_nodes[node] or node in cycle:
            continue

        max1 = float("-inf")
        wh1 = -1
        max2 = float("-inf")
        wh2 = -1

        for node_in_cycle in cycle:
            if score_matrix[node_in_cycle, node] > max1:
                max1 = score_matrix[node_in_cycle, node]
                wh1 = node_in_cycle

            score = cycle_weight + score_matrix[node, node_in_cycle] - score_matrix[parents[node_in_cycle], node_in_cycle]

            if score > max2:
                max2 = score
                wh2 = node_in_cycle

        score_matrix[cycle_representative, node] = max1
        old_input[cycle_representative, node] = old_input[wh1, node]
        old_output[cycle_representative, node] = old_output[wh1, node]
        score_matrix[node, cycle_representative] = max2
        old_output[node, cycle_representative] = old_output[node, wh2]
        old_input[node, cycle_representative] = old_input[node, wh2]

    rep_cons = []
    for i, node_in_cycle in enumerate(cycle):
        rep_cons.append(set())
        for cc in reps[node_in_cycle]:
            rep_cons[i].add(cc)

    # For the next recursive iteration,
    # we want to consider the cycle as a single node.
    # Here we collapse the cycle into the first node
    # in the cycle (first node is arbitrary), set
    # all the other nodes not be considered in the
    # next iteration. 
    for node_in_cycle in cycle[1:]:
        current_nodes[node_in_cycle] = False
        for cc in reps[node_in_cycle]:
            reps[cycle_representative].add(cc)

    chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, reps)

    # Expansion stage.
    # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
    found = False
    wh = -1
    for i, node in enumerate(cycle):
        for repc in rep_cons[i]:
            if repc in final_edges:
                wh = node
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
