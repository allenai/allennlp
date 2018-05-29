from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import torch


def construct_prefix_tree(targets: Union[torch.Tensor, List[List[List[int]]]],
                          target_mask: Optional[torch.Tensor] = None) -> List[Dict[Tuple[int, ...], Set[int]]]:
    """
    Takes a list of valid target action sequences and creates a mapping from all possible
    (valid) action prefixes to allowed actions given that prefix.  While the method is called
    ``construct_prefix_tree``, we're actually returning a map that has as keys the paths to
    `all internal nodes of the trie`, and as values all of the outgoing edges from that node.

    ``targets`` is assumed to be a tensor of shape ``(batch_size, num_valid_sequences,
    sequence_length)``.  If the mask is not ``None``, it is assumed to have the same shape, and
    we will ignore any value in ``targets`` that has a value of ``0`` in the corresponding
    position in the mask.  We assume that the mask has the format 1*0* for each item in
    ``targets`` - that is, once we see our first zero, we stop processing that target.

    For example, if ``targets`` is the following tensor: ``[[1, 2, 3], [1, 4, 5]]``, the return
    value will be: ``{(): set([1]), (1,): set([2, 4]), (1, 2): set([3]), (1, 4): set([5])}``.

    This could be used, e.g., to do an efficient constrained beam search, or to efficiently
    evaluate the probability of all of the target sequences.
    """
    batched_allowed_transitions: List[Dict[Tuple[int, ...], Set[int]]] = []

    if not isinstance(targets, list):
        assert targets.dim() == 3, "targets tensor needs to be batched!"
        targets = targets.detach().cpu().numpy().tolist()
    if target_mask is not None:
        target_mask = target_mask.detach().cpu().numpy().tolist()
    else:
        target_mask = [None for _ in targets]

    for instance_targets, instance_mask in zip(targets, target_mask):
        allowed_transitions: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        for i, target_sequence in enumerate(instance_targets):
            history: Tuple[int, ...] = ()
            for j, action in enumerate(target_sequence):
                if instance_mask and instance_mask[i][j] == 0:
                    break
                allowed_transitions[history].add(action)
                history = history + (action,)
        batched_allowed_transitions.append(allowed_transitions)
    return batched_allowed_transitions
