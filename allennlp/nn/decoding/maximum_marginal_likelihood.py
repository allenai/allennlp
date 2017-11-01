from collections import defaultdict
from typing import Dict, Tuple, Set

import torch

from allennlp.common import Params
from allennlp.nn import util
from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn.decoding.decoder_trainer import DecoderTrainer


@DecoderTrainer.register('max_marginal_likelihood')
class MaximumMarginalLikelihood(DecoderTrainer):
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
               targets: torch.Tensor = None,
               target_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if targets.dim() != 2:
            raise NotImplementedError("This implementation cannot yet handle batched inputs")
        allowed_transitions = self._create_allowed_transitions(targets, target_mask)
        finished_states = []
        states = [initial_state]
        step_num = 0
        while states:
            step_num += 1
            next_states = []
            for state in states:
                allowed_actions = allowed_transitions[tuple(state.action_history)]
                generator = decode_step.take_step(state, allowed_actions)
                while True:
                    try:
                        next_state = next(generator)
                    except StopIteration:
                        break
                    if next_state.is_finished():
                        finished_states.append(next_state)
                    else:
                        next_states.append(next_state)
            states = next_states

        scores = torch.cat([state.score for state in finished_states])
        return {'loss': util.logsumexp(scores)}

    @staticmethod
    def _create_allowed_transitions(targets: torch.Tensor,
                                    target_mask: torch.Tensor = None) -> Dict[Tuple[int, ...], Set[int]]:
        """
        Takes a list of valid target action sequences and creates a mapping from all possible
        (valid) action prefixes to allowed actions given that prefix.  ``targets`` is assumed to be
        a tensor of shape ``(num_valid_sequences, sequence_length)``.  If the mask is not ``None``,
        it is assumed to have the same shape, and we will ignore any value in ``targets`` that has
        a value of ``0`` in the corresponding position in the mask.  We assume that the mask has
        the format 1*0* for each item in ``targets`` - that is, once we see our first zero, we stop
        processing that target.

        Because of some implementation details around creating seq2seq datasets, our targets will
        have a start and end symbol added to them.  We want to ignore the start symbol here,
        because it's not actually a target, it's an input.

        For example, if ``targets`` is the following tensor: ``[[S, 2, 3], [S, 4, 5]]``, the return
        value will be: ``{(): set([2, 4]), (2,): set([3]), (4,): set([5])}`` (note that ``S``,
        which we use to refer to the start symbol, does not appear in the return value).

        We use this to prune the set of actions we consider during decoding, because we only need
        to score the sequences in ``targets``.

        While the example and types above say that this should be a 2D tensor, the implementation
        logic is actually more flexible - any sequence of sequences is valid here, with any
        contained types.  So your targets could be strings, for instance, if your ``DecoderStep``
        and ``DecoderState`` objects know how to handle that.
        """
        allowed_transitions: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        for i, target_sequence in enumerate(targets):
            history: Tuple[int, ...] = ()
            for j, action in enumerate(target_sequence[1:]):
                if target_mask[i][j + 1].data[0] == 0:  # +1 because we're starting at index 1, not 0
                    break
                allowed_transitions[history].add(action.data[0])
                history = history + (action.data[0],)
        return allowed_transitions

    @classmethod
    def from_params(cls, params: Params) -> 'MaximumMarginalLikelihood':
        params.assert_empty(cls.__name__)
        return cls()
