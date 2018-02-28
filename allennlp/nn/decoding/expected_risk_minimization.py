from typing import Dict, List
from collections import defaultdict

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn.decoding.decoder_trainer import DecoderTrainer
from allennlp.nn import util as nn_util


@DecoderTrainer.register('expected_risk_minimization')
class ExpectedRiskMinimization(DecoderTrainer):
    """
    This class implements a trainer that minimizes the expected value of a cost function over the
    space of some candidate sequences produced by a decoder. We generate the candidate sequences by
    performing beam search (which is one of the two popular ways of getting these sequences, the
    other one being sampling; see "Classical Structured Prediction Losses for Sequence to Sequence
    Learning" by Edunov et al., 2017 for more details).
    Note that we do not have a notion of targets here, so we're breaking the API of DecoderTrainer
    a bit.
    """
    def __init__(self, beam_size: int) -> None:
        self._beam_size = beam_size

    def decode(self,  # type: ignore  # ``DecoderTrainer.decode`` also takes targets and their masks
               initial_state: DecoderState,
               decode_step: DecoderStep,
               max_num_steps: int) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        finished_states = []
        states = [initial_state]
        num_steps = 0
        while states and num_steps < max_num_steps:
            next_states = []
            grouped_state = states[0].combine_states(states)
            # These states already come sorted.
            for next_state in decode_step.take_step(grouped_state):
                finished, not_finished = next_state.split_finished()
                if finished is not None:
                    finished_states.append(finished)
                if not_finished is not None:
                    next_states.append(not_finished)

            states = self._prune_beam(next_states)
            num_steps += 1
        loss = 0.0
        finished_model_scores = self._get_model_scores_by_batch(finished_states)
        finished_costs = self._get_costs_by_batch(finished_states)
        for index in finished_model_scores:
            # Finished model scores are log-probabilities of the predicted sequences. We convert
            # them into sequence probabilities and re-normalize them to compute expected cost under
            # the distribution approximated by the beam search.
            costs = torch.cat(finished_costs[index])
            logprobs = torch.cat(finished_model_scores[index])
            # Note that Edunov et al. normalize the logprobs by length of the output sequence before
            # computing exp to get the probability. I think this makes sense for machine translation
            # (which they evaluate on), since predictions should not be discriminated by length. Not
            # sure if it does for semantic parsing.
            # TODO (pradeep): Make length normalization an option.
            renormalized_probs = nn_util.masked_softmax(logprobs, None)
            loss += renormalized_probs.dot(costs)
        return {'loss': loss / len(finished_model_scores),
                'best_action_sequence': self._get_best_action_sequences(finished_states)}

    @staticmethod
    def _get_model_scores_by_batch(states: List[DecoderState]) -> Dict[int, List[Variable]]:
        batch_scores: Dict[int, List[Variable]] = defaultdict(list)
        for state in states:
            for batch_index, model_score in zip(state.batch_indices,
                                                state.score):
                batch_scores[batch_index].append(model_score)
        return batch_scores

    @staticmethod
    def _get_costs_by_batch(states: List[DecoderState]) -> Dict[int, List[Variable]]:
        batch_costs: Dict[int, List[Variable]] = defaultdict(list)
        for state in states:
            cost = state.get_cost()
            # Since this is a finished state, its group size is 1, and we just take the only batch
            # index.
            batch_index = state.batch_indices[0]
            batch_costs[batch_index].append(cost)
        return batch_costs

    @staticmethod
    def _get_best_action_sequences(finished_states: List[DecoderState]) -> Dict[int, List[int]]:
        """
        Returns the best action sequences for each item based on model scores.
        """
        batch_scores: Dict[int, List[Variable]] = defaultdict(list)
        batch_action_histories: Dict[int, List[List[int]]] = defaultdict(list)
        for state in finished_states:
            for batch_index, score, action_history in zip(state.batch_indices,
                                                          state.score,
                                                          state.action_history):
                batch_scores[batch_index].append(score)
                batch_action_histories[batch_index].append(action_history)

        best_action_sequences: Dict[int, List[int]] = {}
        for batch_index, scores in batch_scores.items():
            _, sorted_indices = torch.cat(scores).sort(-1, descending=True)
            cpu_indices = [int(index) for index in sorted_indices.data.cpu().numpy()]
            best_action_sequence = batch_action_histories[batch_index][cpu_indices[0]]
            best_action_sequences[batch_index] = best_action_sequence
        return best_action_sequences

    def _prune_beam(self, states: List[DecoderState]) -> List[DecoderState]:
        """
        Prunes a beam, and keeps at most ``self._beam_size`` states per instance. We
        assume that the ``states`` are grouped, with a group size of 1, and that they're already
        sorted.
        """
        num_states_per_instance: Dict[int, int] = defaultdict(int)
        pruned_states = []
        for state in states:
            assert len(state.batch_indices) == 1
            batch_index = state.batch_indices[0]
            if num_states_per_instance[batch_index] < self._beam_size:
                pruned_states.append(state)
                num_states_per_instance[batch_index] += 1
        return pruned_states

    @classmethod
    def from_params(cls, params: Params) -> 'ExpectedRiskMinimization':
        beam_size = params.pop('beam_size')
        params.assert_empty(cls.__name__)
        return cls(beam_size)
