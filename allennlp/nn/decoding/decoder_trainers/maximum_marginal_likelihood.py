from collections import defaultdict
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch.autograd import Variable

from allennlp.nn import util
from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn.decoding.decoder_trainers.decoder_trainer import DecoderTrainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MaximumMarginalLikelihood(DecoderTrainer[Tuple[torch.Tensor, torch.Tensor]]):
    """
    This class trains a decoder by maximizing the marginal likelihood of the targets.  That is,
    during training, we are given a `set` of acceptable or possible target sequences, and we
    optimize the `sum` of the probability the model assigns to each item in the set.  This allows
    the model to distribute its probability mass over the set however it chooses, without forcing
    `all` of the given target sequences to have high probability.  This is helpful, for example, if
    you have good reason to expect that the correct target sequence is in the set, but aren't sure
    `which` of the sequences is actually correct.

    This implementation of maximum marginal likelihood requires the model you use to be `locally
    normalized`; that is, at each decoding timestep, we assume that the model creates a normalized
    probability distribution over actions.  This assumption is necessary, because we do no explicit
    normalization in our loss function, we just sum the probabilities assigned to all correct
    target sequences, relying on the local normalization at each time step to push probability mass
    from bad actions to good ones.
    """
    def decode(self,
               initial_state: DecoderState,
               decode_step: DecoderStep,
               supervision: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        targets, target_mask = supervision
        allowed_transitions = self._create_allowed_transitions(targets, target_mask)
        finished_states = []
        states = [initial_state]
        step_num = 0
        while states:
            step_num += 1
            next_states = []
            # We group together all current states to get more efficient (batched) computation.
            grouped_state = states[0].combine_states(states)
            allowed_actions = self._get_allowed_actions(grouped_state, allowed_transitions)
            # This will store a set of (batch_index, action_history) tuples, and we'll check it
            # against the allowed actions to make sure we're actually scoring all of the actions we
            # are supposed to.
            actions_taken: Set[Tuple[int, Tuple[int, ...]]] = set()
            for next_state in decode_step.take_step(grouped_state, allowed_actions=allowed_actions):
                actions_taken.add((next_state.batch_indices[0], tuple(next_state.action_history[0])))
                if next_state.is_finished():
                    finished_states.append(next_state)
                else:
                    next_states.append(next_state)
            states = next_states
            self._check_all_actions_taken(actions_taken, grouped_state, allowed_actions)

        # This is a dictionary of lists - for each batch instance, we want the score of all
        # finished states.  So this has shape (batch_size, num_target_action_sequences), though
        # it's not actually a tensor, because different batch instance might have different numbers
        # of finished states.
        batch_scores = self._group_scores_by_batch(finished_states)
        loss = 0
        for scores in batch_scores.values():  # we don't care about the batch index, just the scores
            loss += -util.logsumexp(torch.cat(scores))
        return {'loss': loss / len(batch_scores)}

    @staticmethod
    def _create_allowed_transitions(targets: Union[torch.Tensor, List[List[List[int]]]],
                                    target_mask: Optional[torch.Tensor] = None) -> List[Dict[Tuple[int, ...],
                                                                                             Set[int]]]:
        """
        Takes a list of valid target action sequences and creates a mapping from all possible
        (valid) action prefixes to allowed actions given that prefix.  ``targets`` is assumed to be
        a tensor of shape ``(batch_size, num_valid_sequences, sequence_length)``.  If the mask is
        not ``None``, it is assumed to have the same shape, and we will ignore any value in
        ``targets`` that has a value of ``0`` in the corresponding position in the mask.  We assume
        that the mask has the format 1*0* for each item in ``targets`` - that is, once we see our
        first zero, we stop processing that target.

        For example, if ``targets`` is the following tensor: ``[[1, 2, 3], [1, 4, 5]]``, the return
        value will be: ``{(): set([1]), (1,): set([2, 4]), (1, 2): set([3]), (1, 4): set([5])}``.

        We use this to prune the set of actions we consider during decoding, because we only need
        to score the sequences in ``targets``.
        """
        batched_allowed_transitions: List[Dict[Tuple[int, ...], Set[int]]] = []

        if not isinstance(targets, list):
            assert targets.dim() == 3, "targets tensor needs to be batched!"
            targets = targets.data.cpu().numpy().tolist()
        if target_mask is not None:
            target_mask = target_mask.data.cpu().numpy().tolist()
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

    @staticmethod
    def _check_all_actions_taken(action_histories: Set[Tuple[int, Tuple[int, ...]]],
                                 grouped_state: DecoderState,
                                 allowed_actions: List[Set[int]]) -> None:
        expected_histories = set()
        for i, batch_index in enumerate(grouped_state.batch_indices):
            action_history = grouped_state.action_history[i]
            for allowed_action in allowed_actions[i]:
                expected_histories.add((batch_index, tuple(action_history + [allowed_action])))
        assert action_histories == expected_histories, f'Expected: {expected_histories}'

    @staticmethod
    def _get_allowed_actions(state: DecoderState,
                             allowed_transitions: List[Dict[Tuple[int, ...], Set[int]]]) -> List[Set[int]]:
        """
        Takes a list of allowed transitions for each element of a batch, and a decoder state that
        contains the current action history for each `group` element, and returns a list of allowed
        actions in the current state, also for each `group` element.
        """
        allowed_actions = []
        for batch_index, action_history in zip(state.batch_indices, state.action_history):
            allowed_actions.append(allowed_transitions[batch_index][tuple(action_history)])
        return allowed_actions

    @staticmethod
    def _group_scores_by_batch(finished_states: List[DecoderState]) -> Dict[int, List[Variable]]:
        """
        Takes a list of finished states and groups all final scores for each batch element into a
        list.  This is not trivial because the instances in the batch all might "finish" at
        different times, so we re-batch them during the training process.  We need to recover the
        original batch grouping so we can compute the loss correctly.
        """
        batch_scores: Dict[int, List[Variable]] = defaultdict(list)
        for state in finished_states:
            for score, batch_index in zip(state.score, state.batch_indices):
                batch_scores[batch_index].append(score)
        return batch_scores
