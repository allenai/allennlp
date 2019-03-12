from collections import defaultdict
from typing import Dict, Generic, List, TypeVar, Tuple

import torch

from allennlp.common.registrable import FromParams
from allennlp.state_machines import util
from allennlp.state_machines.states import State
from allennlp.state_machines.transition_functions import TransitionFunction

StateType = TypeVar('StateType', bound=State)  # pylint: disable=invalid-name


class BeamSearch(FromParams, Generic[StateType]):
    """
    This class implements beam search over transition sequences given an initial ``State`` and a
    ``TransitionFunction``, returning the highest scoring final states found by the beam (the
    states will keep track of the transition sequence themselves).

    The initial ``State`` is assumed to be `batched`.  The value we return from the search is a
    dictionary from batch indices to ranked finished states.

    IMPORTANT: We assume that the ``TransitionFunction`` that you are using returns possible next
    states in sorted order, so we do not do an additional sort inside of ``BeamSearch.search()``.
    If you're implementing your own ``TransitionFunction``, you must ensure that you've sorted the
    states that you return.

    Parameters
    ----------
    beam_size : ``int``
        The beam size to use.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to `beam_size`. Setting this parameter
        to a number smaller than `beam_size` may give better results, as it can introduce
        more diversity into the search. See Freitag and Al-Onaizan 2017,
        "Beam Search Strategies for Neural Machine Translation".
    initial_sequence : ``torch.Tensor``, optional (default = None)
        If you provide a (sequence_length,) tensor here, the beam search will be constrained
        to only sequences that begin with the provided initial_sequence.
    keep_beam_details : ``bool``, optional (default = False)
        If True, we store snapshots of each beam in an instance variable ``beam_snapshots``,
        which is a dict: { batch_index -> [timestep0_histories, ..., timestepk_histories] },
        where a "timestep history" is just a pair (score, action_history) that was considered
        at that timestep.
    """
    def __init__(self,
                 beam_size: int,
                 per_node_beam_size: int = None,
                 initial_sequence: torch.Tensor = None,
                 keep_beam_details: bool = False) -> None:
        self._beam_size = beam_size
        self._per_node_beam_size = per_node_beam_size or beam_size

        if initial_sequence is not None:
            # construct_prefix_tree wants a tensor of shape (batch_size, num_sequences, sequence_length)
            # so we need to add the first two dimensions in. This returns a list, but we're assuming
            # batch size 1, so we extract the first element.
            self._allowed_transitions = util.construct_prefix_tree(initial_sequence.view(1, 1, -1))[0]
        else:
            self._allowed_transitions = None

        if keep_beam_details:
            # mapping from batch_index to a list (timesteps) of lists (beam elements)
            # of pairs (score, action_history)
            self.beam_snapshots: Dict[int, List[List[Tuple[float, List[int]]]]] = {}
        else:
            self.beam_snapshots = None

    def constrained_to(self, initial_sequence: torch.Tensor, keep_beam_details: bool = True) -> 'BeamSearch':
        """
        Return a new BeamSearch instance that's like this one but with the specified constraint.
        """
        return BeamSearch(self._beam_size, self._per_node_beam_size, initial_sequence, keep_beam_details)

    def search(self,
               num_steps: int,
               initial_state: StateType,
               transition_function: TransitionFunction,
               keep_final_unfinished_states: bool = True) -> Dict[int, List[StateType]]:
        """
        Parameters
        ----------
        num_steps : ``int``
            How many steps should we take in our search?  This is an upper bound, as it's possible
            for the search to run out of valid actions before hitting this number, or for all
            states on the beam to finish.
        initial_state : ``StateType``
            The starting state of our search.  This is assumed to be `batched`, and our beam search
            is batch-aware - we'll keep ``beam_size`` states around for each instance in the batch.
        transition_function : ``TransitionFunction``
            The ``TransitionFunction`` object that defines and scores transitions from one state to the
            next.
        keep_final_unfinished_states : ``bool``, optional (default=True)
            If we run out of steps before a state is "finished", should we return that state in our
            search results?

        Returns
        -------
        best_states : ``Dict[int, List[StateType]]``
            This is a mapping from batch index to the top states for that instance.
        """
        finished_states: Dict[int, List[StateType]] = defaultdict(list)
        states = [initial_state]
        step_num = 1

        # Erase stored beams, if we're tracking them.
        if self.beam_snapshots is not None:
            self.beam_snapshots = defaultdict(list)

        while states and step_num <= num_steps:
            next_states: Dict[int, List[StateType]] = defaultdict(list)
            grouped_state = states[0].combine_states(states)

            if self._allowed_transitions:
                # We were provided an initial sequence, so we need to check
                # if the current sequence is still constrained.
                key = tuple(grouped_state.action_history[0])
                if key in self._allowed_transitions:
                    # We're still in the initial_sequence, so our hand is forced.
                    allowed_actions = [self._allowed_transitions[key]]
                else:
                    # We've gone past the end of the initial sequence, so no constraint.
                    allowed_actions = None
            else:
                # No initial sequence was provided, so all actions are allowed.
                allowed_actions = None

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
                    if step_num == num_steps and keep_final_unfinished_states:
                        finished_states[batch_index].append(next_state)
                    next_states[batch_index].append(next_state)
            states = []
            for batch_index, batch_states in next_states.items():
                # The states from the generator are already sorted, so we can just take the first
                # ones here, without an additional sort.
                states.extend(batch_states[:self._beam_size])

                if self.beam_snapshots is not None:
                    # Add to beams
                    self.beam_snapshots[batch_index].append(
                            [(state.score[0].item(), state.action_history[0])
                             for state in batch_states]
                    )
            step_num += 1

        # Add finished states to the stored beams as well.
        if self.beam_snapshots is not None:
            for batch_index, states in finished_states.items():
                for state in states:
                    score = state.score[0].item()
                    action_history = state.action_history[0]

                    while len(self.beam_snapshots[batch_index]) < len(action_history):
                        self.beam_snapshots[batch_index].append([])

                    self.beam_snapshots[batch_index][len(action_history) - 1].append((score, action_history))

        best_states: Dict[int, List[StateType]] = {}
        for batch_index, batch_states in finished_states.items():
            # The time this sort takes is pretty negligible, no particular need to optimize this
            # yet.  Maybe with a larger beam size...
            finished_to_sort = [(-state.score[0].item(), state) for state in batch_states]
            finished_to_sort.sort(key=lambda x: x[0])
            best_states[batch_index] = [state[1] for state in finished_to_sort[:self._beam_size]]
        return best_states
