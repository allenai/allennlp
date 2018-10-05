from collections import defaultdict
from typing import Dict, List, Optional

import torch

from allennlp.state_machines import util
from allennlp.state_machines.states import State
from allennlp.state_machines.transition_functions import TransitionFunction


class ConstrainedBeamSearch:
    """
    This class implements beam search over transition sequences given an initial ``State``, a
    ``TransitionFunction``, and a list of allowed transition sequences.  We will do a beam search
    `over the list of allowed sequences` and return the highest scoring states found by the beam.
    This is only actually a `beam search` if your beam size is smaller than the list of allowed
    transition sequences; otherwise, we are just scoring and sorting the sequences using a prefix
    tree.

    The initial ``State`` is assumed to be `batched`.  The value we return from the search is a
    dictionary from batch indices to ranked finished states.

    IMPORTANT: We assume that the ``TransitionFunction`` that you are using returns possible next
    states in sorted order, so we do not do an additional sort inside of
    ``ConstrainedBeamSearch.search()``.  If you're implementing your own ``TransitionFunction``,
    you must ensure that you've sorted the states that you return.

    Parameters
    ----------
    beam_size : ``Optional[int]``
        The beam size to use.  Because this is a `constrained` beam search, we allow for the case
        where you just want to evaluate all options in the constrained set.  In that case, you
        don't need a beam, and you can pass a beam size of ``None``, and we will just evaluate
        everything.  This lets us be more efficient in :func:`TransitionFunction.take_step` and
        skip the sorting that is typically done there.
    allowed_sequences : ``torch.Tensor``
        A ``(batch_size, num_sequences, sequence_length)`` tensor containing the transition
        sequences that we will search in.  The values in this tensor must match whatever the
        ``State`` keeps in its ``action_history`` variable (typically this is action indices).
    allowed_sequence_mask : ``torch.Tensor``
        A ``(batch_size, num_sequences, sequence_length)`` tensor indicating whether each entry in
        the ``allowed_sequences`` tensor is padding.  The allowed sequences could be padded both on
        the ``num_sequences`` dimension and the ``sequence_length`` dimension.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to `beam_size`. Setting this parameter
        to a number smaller than `beam_size` may give better results, as it can introduce
        more diversity into the search. See Freitag and Al-Onaizan 2017,
        "Beam Search Strategies for Neural Machine Translation".
    """
    def __init__(self,
                 beam_size: Optional[int],
                 allowed_sequences: torch.Tensor,
                 allowed_sequence_mask: torch.Tensor,
                 per_node_beam_size: int = None) -> None:
        self._beam_size = beam_size
        self._per_node_beam_size = per_node_beam_size or beam_size
        self._allowed_transitions = util.construct_prefix_tree(allowed_sequences, allowed_sequence_mask)

    def search(self,
               initial_state: State,
               transition_function: TransitionFunction) -> Dict[int, List[State]]:
        """
        Parameters
        ----------
        initial_state : ``State``
            The starting state of our search.  This is assumed to be `batched`, and our beam search
            is batch-aware - we'll keep ``beam_size`` states around for each instance in the batch.
        transition_function : ``TransitionFunction``
            The ``TransitionFunction`` object that defines and scores transitions from one state to the
            next.

        Returns
        -------
        best_states : ``Dict[int, List[State]]``
            This is a mapping from batch index to the top states for that instance.
        """
        finished_states: Dict[int, List[State]] = defaultdict(list)
        states = [initial_state]
        step_num = 0
        while states:
            step_num += 1
            next_states: Dict[int, List[State]] = defaultdict(list)
            grouped_state = states[0].combine_states(states)
            allowed_actions = []
            for batch_index, action_history in zip(grouped_state.batch_indices,
                                                   grouped_state.action_history):
                allowed_actions.append(self._allowed_transitions[batch_index][tuple(action_history)])
            for next_state in transition_function.take_step(grouped_state,
                                                            max_actions=self._per_node_beam_size,
                                                            allowed_actions=allowed_actions):
                # NOTE: we're doing state.batch_indices[0] here (and similar things below),
                # hard-coding a group size of 1.  But, our use of `next_state.is_finished()`
                # already checks for that, as it crashes if the group size is not 1.
                batch_index = next_state.batch_indices[0]
                if next_state.is_finished():
                    finished_states[batch_index].append(next_state)
                else:
                    next_states[batch_index].append(next_state)
            states = []
            for batch_index, batch_states in next_states.items():
                # The states from the generator are already sorted, so we can just take the first
                # ones here, without an additional sort.
                if self._beam_size:
                    batch_states = batch_states[:self._beam_size]
                states.extend(batch_states)
        best_states: Dict[int, List[State]] = {}
        for batch_index, batch_states in finished_states.items():
            # The time this sort takes is pretty negligible, no particular need to optimize this
            # yet.  Maybe with a larger beam size...
            finished_to_sort = [(-state.score[0].item(), state) for state in batch_states]
            finished_to_sort.sort(key=lambda x: x[0])
            best_states[batch_index] = [state[1] for state in finished_to_sort[:self._beam_size]]
        return best_states
