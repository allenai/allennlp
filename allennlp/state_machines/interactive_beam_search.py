from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from allennlp.state_machines import util
from allennlp.state_machines.beam_search import BeamSearch, StateType
from allennlp.state_machines.transition_functions import TransitionFunction


class InteractiveBeamSearch(BeamSearch[StateType]):
    """
    This class is designed to be a drop-in-replacement for `BeamSearch` that allows you to
    specify the initial realization of the sequence. The idea is that in an interactive
    setting you can force the beam search down certain paths, making for interesting
    interactive demos.

    Although the underlying BeamSearch infrastructure is batched, here we assume that the
    batch has only a single element.

    IMPORTANT: We assume that the ``TransitionFunction`` that you are using returns possible next
    states in sorted order, so we do not do an additional sort inside of
    ``InteractiveBeamSearch.search()``.  If you're implementing your own ``TransitionFunction``,
    you must ensure that you've sorted the states that you return.

    Parameters
    ----------
    beam_size : int
        The beam size to use.
    initial_sequence : ``torch.Tensor``, optional (default = None)
        An ``(initial_sequence_length,)`` tensor containing the start of the sequence.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to `beam_size`. Setting this parameter
        to a number smaller than `beam_size` may give better results, as it can introduce
        more diversity into the search. See Freitag and Al-Onaizan 2017,
        "Beam Search Strategies for Neural Machine Translation".
    """
    def __init__(self,
                 beam_size: int,
                 initial_sequence: torch.Tensor = None,
                 per_node_beam_size: int = None) -> None:
        super().__init__(beam_size, per_node_beam_size)
        if initial_sequence is not None:
            # construct_prefix_tree wants a tensor of shape (batch_size, num_sequences, sequence_length)
            # so we need to add the first two dimensions in. This returns a list, but we're assuming
            # batch size 1, so we extract the first element.
            self._allowed_transitions = util.construct_prefix_tree(initial_sequence.view(1, 1, -1))[0]
        else:
            self._allowed_transitions = {}

        self.beams: List[List[Tuple[float, List[int]]]] = []

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
        step_num = 0
        self.beams = []

        while states and step_num < num_steps:
            step_num += 1
            next_states: Dict[int, List[StateType]] = defaultdict(list)
            grouped_state = states[0].combine_states(states)

            # The only possible constraint on allowed actions is that we're still following
            # the specified initial sequence, which we can check by seeing if the first
            # action history appears in our allowed transitions.
            next_actions = self._allowed_transitions.get(tuple(grouped_state.action_history[0]))

            if next_actions is None:
                allowed_actions = None
            else:
                allowed_actions = [next_actions]

            for next_state in transition_function.take_step(grouped_state,
                                                            max_actions=self._per_node_beam_size,
                                                            allowed_actions=allowed_actions):
                # NOTE: we're doing state.batch_indices[0] here (and similar things below),
                # hard-coding a group size of 1.  But, our use of `next_state.is_finished()`
                # already checks for that, as it crashes if the group size is not 1.
                batch_index = next_state.batch_indices[0]
                is_finished = next_state.is_finished()
                if is_finished or (step_num == num_steps and keep_final_unfinished_states):
                    finished_states[batch_index].append(next_state)
                if not is_finished:
                    next_states[batch_index].append(next_state)
            states = []
            for batch_index, batch_states in next_states.items():
                # The states from the generator are already sorted, so we can just take the first
                # ones here, without an additional sort.
                if self._beam_size:
                    batch_states = batch_states[:self._beam_size]
                states.extend(batch_states)

                # Add to beams
                self.beams.append([(state.score[0].item(), state.action_history[0])
                                   for state in batch_states])

        # Add finished states to the beams as well.
        for state in finished_states[0]:
            score = state.score[0].item()
            action_history = state.action_history[0]

            while len(self.beams) < len(action_history):
                self.beams.append([])

            self.beams[len(action_history) - 1].append((score, action_history))


        best_states: Dict[int, List[StateType]] = {}
        for batch_index, batch_states in finished_states.items():
            # The time this sort takes is pretty negligible, no particular need to optimize this
            # yet.  Maybe with a larger beam size...
            finished_to_sort = [(-state.score[0].item(), state) for state in batch_states]
            finished_to_sort.sort(key=lambda x: x[0])
            best_states[batch_index] = [state[1] for state in finished_to_sort[:self._beam_size]]
        return best_states
