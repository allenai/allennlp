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
    max_num_decoded_sequences : ``int``, optional (default=1)
        Maximum number of sorted decoded sequences to return. Defaults to 1.
    max_num_finished_states : ``int``, optional (default = None)
        Maximum number of finished states to keep after search. This is to finished states as
        ``beam_size`` is to unfinished ones. Costs are computed for only these number of states per
        instance. If not set, we will keep all the finished states.
    """
    def __init__(self,
                 beam_size: int,
                 normalize_by_length: bool,
                 max_decoding_steps: int,
                 max_num_decoded_sequences: int = 1,
                 max_num_finished_states: int = None) -> None:
        self._beam_size = beam_size
        self._normalize_by_length = normalize_by_length
        self._max_decoding_steps = max_decoding_steps
        self._max_num_decoded_sequences = max_num_decoded_sequences
        self._max_num_finished_states = max_num_finished_states

    def decode(self,
               initial_state: DecoderState,
               decode_step: DecoderStep,
               supervision: Callable[[StateType], torch.Tensor]) -> Dict[str, torch.Tensor]:
        cost_function = supervision
        finished_states = self._get_finished_states(initial_state, decode_step)
        loss = Variable(initial_state.score[0].data.new([0.0]))
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
                'best_action_sequences': self._get_best_action_sequences(finished_states)}

    def _get_finished_states(self,
                             initial_state: DecoderState,
                             decode_step: DecoderStep) -> List[StateType]:
        finished_states = []
        states = [initial_state]
        num_steps = 0
        while states and num_steps < self._max_decoding_steps:
            next_states = []
            grouped_state = states[0].combine_states(states)
            # These states already come sorted.
            for next_state in decode_step.take_step(grouped_state):
                if next_state.is_finished():
                    finished_states.append(next_state)
                else:
                    next_states.append(next_state)

            states = self._prune_beam(states=next_states,
                                      beam_size=self._beam_size,
                                      sort_states=False)
            num_steps += 1
        if self._max_num_finished_states is not None:
            finished_states = self._prune_beam(states=finished_states,
                                               beam_size=self._max_num_finished_states,
                                               sort_states=True)
        return finished_states

    # TODO(pradeep): Move this method to nn.decoding.util
    @staticmethod
    def _prune_beam(states: List[DecoderState],
                    beam_size: int,
                    sort_states: bool = False) -> List[DecoderState]:
        """
        This method can be used to prune the set of unfinished states on a beam or finished states
        at the end of search. In the former case, the states need not be sorted because the all come
        from the same decoding step, which does the sorting. However, if the states are finished and
        this method is called at the end of the search, they need to be sorted because they come
        from different decoding steps.
        """
        states_by_batch_index: Dict[int, List[DecoderState]] = defaultdict(list)
        for state in states:
            assert len(state.batch_indices) == 1
            batch_index = state.batch_indices[0]
            states_by_batch_index[batch_index].append(state)
        pruned_states = []
        for _, instance_states in states_by_batch_index.items():
            if sort_states:
                scores = torch.cat([state.score[0] for state in instance_states])
                _, sorted_indices = scores.sort(-1, descending=True)
                sorted_states = [instance_states[i] for i in sorted_indices.data.cpu().numpy()]
                instance_states = sorted_states
            for state in instance_states[:beam_size]:
                pruned_states.append(state)
        return pruned_states

    def _get_model_scores_by_batch(self, states: List[StateType]) -> Dict[int, List[Variable]]:
        batch_scores: Dict[int, List[Variable]] = defaultdict(list)
        for state in states:
            for batch_index, model_score, history in zip(state.batch_indices,
                                                         state.score,
                                                         state.action_history):
                if self._normalize_by_length:
                    path_length = Variable(model_score.data.new([len(history)]))
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

    def _get_best_action_sequences(self,
                                   finished_states: List[StateType]) -> Dict[int, List[List[int]]]:
        """
        Returns the best action sequences for each item based on model scores. We return at most
        ``self._max_num_decoded_sequences`` number of sequences per instance.
        """
        batch_action_histories: Dict[int, List[List[int]]] = defaultdict(list)
        for state in finished_states:
            for batch_index, action_history in zip(state.batch_indices,
                                                   state.action_history):
                batch_action_histories[batch_index].append(action_history)

        batch_scores = self._get_model_scores_by_batch(finished_states)
        best_action_sequences: Dict[int, List[List[int]]] = {}
        for batch_index, scores in batch_scores.items():
            _, sorted_indices = torch.cat(scores).sort(-1, descending=True)
            cpu_indices = [int(index) for index in sorted_indices.data.cpu().numpy()]
            best_action_indices = cpu_indices[:self._max_num_decoded_sequences]
            instance_best_sequences = [batch_action_histories[batch_index][i]
                                       for i in best_action_indices]
            best_action_sequences[batch_index] = instance_best_sequences
        return best_action_sequences
