from collections import defaultdict
from typing import Dict, Tuple, Set

import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.nn import util
from allennlp.nn.decoding.decode_step import DecodeStep
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn.decoding.decoding_algorithm import DecodingAlgorithm


@DecodingAlgorithm.register('max_marginal_likelihood')
class MaximumMarginalLikelihoodDecoder(DecodingAlgorithm):
    """
    This class implements maximum marginal likelihood decoding.  That is, during training, we are
    given a `set` of acceptable or possible target sequences, and we optimize the `sum` of the
    probability the model assigns to each item in the set.  This allows the model to distribute its
    probability mass over the set however it chooses, without forcing `all` of the given target
    sequences to have high probability.  This is helpful, for example, if you have good reason to
    expect that the correct target sequence is in the set, but aren't sure `which` of the sequences
    is actually correct.

    This implementation of maximum marginal likelihood requires the model you use to be `locally
    normalized`; that is, at each decoding timestep, we assume that the model creates a normalized
    probability distribution over actions.  This assumption is necessary, because we do no explicit
    normalization in our loss function, we just sum the probabilities assigned to all correct
    target sequences, relying on the local normalization at each time step to push probability mass
    from bad actions to good ones.

    Parameters
    ----------
    vocab : ``Vocabulary``
        We need this so that we know the index of the start and end symbols for decoding.
    vocab_namespace : ``str``
        This tells us what namespace to look in to find the index of the start and end symbols.
    scheduled_sampling_ratio : ``float``, optional (default = 0.0)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al., 2015.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 vocab_namespace: str) -> None:
        super(self, MaximumMarginalLikelihoodDecoder).__init__(vocab, vocab_namespace)

    def decode(self,
               num_steps: int,
               initial_state: DecoderState,
               decode_step: DecodeStep,
               targets: torch.Tensor = None,
               target_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if targets.dim() != 2:
            raise NotImplementedError("This implementation cannot yet handle batched inputs")
        allowed_transitions = self._create_allowed_transitions(targets)
        finished_states = []
        states = [initial_state]
        while states:
            next_states = []
            for state in states:
                decoder_input = state.action_history[-1] if state.action_history else self._start_index
                allowed_actions = allowed_transitions[state.action_history]
                for next_state in decode_step.take_step(state, decoder_input, allowed_actions):
                    if next_state.action_history[-1] == self._end_index:
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
        contained types.  So your targets could be strings, for instance, if your ``DecodeStep``
        and ``DecoderState`` objects know how to handle that.
        """
        allowed_transitions: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        for i, target_sequence in enumerate(targets):
            history: Tuple[int, ...] = ()
            for j, action in enumerate(target_sequence[1:]):
                if target_mask[i][j + 1] == 0:  # +1 because we're starting at index 1, not 0
                    break
                allowed_transitions[history].add(action)
                history = history + (action,)
        return allowed_transitions

    @classmethod
    def from_params(cls,
                    vocab: Vocabulary,
                    vocab_namespace: str,
                    params: Params) -> 'MaximumMarginalLikelihoodDecoder':
        return cls(vocab=vocab,
                   vocab_namespace=vocab_namespace)
