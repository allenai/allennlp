from typing import Callable, Dict, List, TypeVar
from collections import defaultdict

import torch
from torch.autograd import Variable

from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn.decoding.decoder_trainers.decoder_trainer import DecoderTrainer
from allennlp.nn import util as nn_util

StateType = TypeVar('StateType', bound=DecoderState)  # pylint: disable=invalid-name


class ExpectedRiskMinimization(DecoderTrainer[Callable[[StateType], torch.Tensor]]):
    """
    This class implements a trainer that minimizes the expected value of a cost function over the
    space of some candidate sequences produced by a decoder. We generate the candidate sequences by
    performing beam search (which is one of the two popular ways of getting these sequences, the
    other one being sampling; see "Classical Structured Prediction Losses for Sequence to Sequence
    Learning" by Edunov et al., 2017 for more details).
    Note that we do not have a notion of targets here, so we're breaking the API of DecoderTrainer
    a bit.

    Parameters
    ----------
    beam_size : ``int``
    noramlize_by_length : ``bool``
        Should the log probabilities be normalized by length before renormalizing them? Edunov et
        al. do this in their work.
    max_decoding_steps : ``int``
        The maximum number of steps we should take during decoding.
    """
    def __init__(self, beam_size: int, normalize_by_length: bool, max_decoding_steps: int) -> None:
        self._beam_size = beam_size
        self._normalize_by_length = normalize_by_length
        self.max_decoding_steps = max_decoding_steps

    def decode(self,
               initial_state: DecoderState,
               decode_step: DecoderStep,
               supervision: Callable[[StateType], torch.Tensor]) -> Dict[str, torch.Tensor]:
        cost_function = supervision
        finished_states = self._get_finished_states(initial_state, decode_step)
        loss = nn_util.new_variable_with_data(initial_state.score[0], torch.Tensor([0.0]))
        finished_model_scores = self._get_model_scores_by_batch(finished_states)
        finished_costs = self._get_costs_by_batch(finished_states, cost_function)
        for batch_index in finished_model_scores:
            # Finished model scores are log-probabilities of the predicted sequences. We convert
            # log probabilities into probabilities and re-normalize them to compute expected cost under
            # the distribution approximated by the beam search.
            costs = torch.cat(finished_costs[batch_index])
            logprobs = torch.cat(finished_model_scores[batch_index])
            # Unmasked softmax of log probabilities will convert them into probabilities and
            # renormalize them.
            renormalized_probs = nn_util.masked_softmax(logprobs, None)
            loss += renormalized_probs.dot(costs)
        mean_loss = loss / len(finished_model_scores)
        return {'loss': mean_loss,
                'best_action_sequence': self._get_best_action_sequences(finished_states)}

    def _get_finished_states(self,
                             initial_state: DecoderState,
                             decode_step: DecoderStep) -> List[StateType]:
        finished_states = []
        states = [initial_state]
        num_steps = 0
        while states and num_steps < self.max_decoding_steps:
            next_states = []
            grouped_state = states[0].combine_states(states)
            # These states already come sorted.
            for next_state in decode_step.take_step(grouped_state):
                if next_state.is_finished():
                    finished_states.append(next_state)
                else:
                    next_states.append(next_state)

            states = self._prune_beam(next_states)
            num_steps += 1
        return finished_states

    def _get_model_scores_by_batch(self, states: List[StateType]) -> Dict[int, List[Variable]]:
        batch_scores: Dict[int, List[Variable]] = defaultdict(list)
        for state in states:
            for batch_index, model_score, history in zip(state.batch_indices,
                                                         state.score,
                                                         state.action_history):
                if self._normalize_by_length:
                    path_length = nn_util.new_variable_with_data(model_score,
                                                                 torch.Tensor([len(history)]))
                    model_score = model_score / path_length
                batch_scores[batch_index].append(model_score)
        return batch_scores

    @staticmethod
    def _get_costs_by_batch(states: List[StateType],
                            cost_function: Callable[[StateType], torch.Tensor]) -> Dict[int, List[Variable]]:
        batch_costs: Dict[int, List[Variable]] = defaultdict(list)
        for state in states:
            cost = cost_function(state)
            # Since this is a finished state, its group size is 1, and we just take the only batch
            # index.
            batch_index = state.batch_indices[0]
            batch_costs[batch_index].append(cost)
        return batch_costs

    def _get_best_action_sequences(self, finished_states: List[StateType]) -> Dict[int, List[int]]:
        """
        Returns the best action sequences for each item based on model scores.
        """
        batch_action_histories: Dict[int, List[List[int]]] = defaultdict(list)
        for state in finished_states:
            for batch_index, action_history in zip(state.batch_indices,
                                                   state.action_history):
                batch_action_histories[batch_index].append(action_history)

        batch_scores = self._get_model_scores_by_batch(finished_states)
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
