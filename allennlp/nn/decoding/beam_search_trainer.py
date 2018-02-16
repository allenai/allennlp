from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn.decoding.decoder_trainer import DecoderTrainer


@DecoderTrainer.register('beam_search')
class BeamSearchTrainer(DecoderTrainer):
    """
    This class implements a trainer that performs beam search and maximizes the scores of the
    finished states in the beam. This is like Beam Search Optimiziation (Wiseman and Rush, 2016),
    except that the loss is not margin based. Note that we do not have a notion of targets here, so
    we're breaking the API of DecoderTrainer a bit.
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

        for state in finished_states:
            state.denotation_is_correct()
        correct_batch_scores, incorrect_batch_scores = self._group_scores_by_batch(finished_states)
        loss = 0
        all_batch_indices = set(correct_batch_scores.keys()).union(incorrect_batch_scores.keys())
        for batch_index in all_batch_indices:
            correct_score = None
            incorrect_score = None
            if batch_index in correct_batch_scores:
                correct_score = torch.max(torch.cat(correct_batch_scores[batch_index]))

            if batch_index in incorrect_batch_scores:
                incorrect_score = torch.max(torch.cat(incorrect_batch_scores[batch_index]))
            # TODO (pradeep): Is 1 the right margin here?
            if correct_score is None:
                loss += (1 + incorrect_score)
            elif incorrect_score is None:
                loss += (1 - correct_score)
            else:
                loss += (1 - correct_score + incorrect_score)

        return {'loss': loss / len(all_batch_indices),
                'best_action_sequence': self._get_best_action_sequences(finished_states)}

    @staticmethod
    def _group_scores_by_batch(finished_states: List[DecoderState]) -> Dict[int,
                                                                            Tuple[List[Variable],
                                                                                  List[Variable]]]:
        # We separate the action histories and scores of action sequences by batch indices.
        correct_batch_scores: Dict[int, List[Variable]] = defaultdict(list)
        incorrect_batch_scores: Dict[int, List[Variable]] = defaultdict(list)
        for state in finished_states:
            denotations_are_correct = state.denotation_is_correct()
            for score, batch_index, is_correct in zip(state.score, state.batch_indices,
                                                      denotations_are_correct):
                if is_correct:
                    correct_batch_scores[batch_index].append(score)
                else:
                    incorrect_batch_scores[batch_index].append(score)
        return correct_batch_scores, incorrect_batch_scores

    @staticmethod
    def _get_best_action_sequences(finished_states: List[DecoderState]) -> Dict[int, List[int]]:
        batch_scores: Dict[int, List[Variable]] = defaultdict(list)
        batch_action_histories: Dict[int, List[List[int]]] = defaultdict(list)
        for state in finished_states:
            for batch_index, score, action_history in zip(state.batch_indices, state.score,
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
    def from_params(cls, params: Params) -> 'BeamSearchTrainer':
        beam_size = params.pop('beam_size')
        params.assert_empty(cls.__name__)
        return cls(beam_size)
