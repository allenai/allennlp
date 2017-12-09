from typing import Dict, List
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
    This class implements a trainer that performs beam search and maximizes the scores of the finished states
    in the beam. This is like Beam Search Optimiziation (Wiseman and Rush, 2016), except that the loss
    is not margin based. Not that we do not have a notion of targets here, so we're breaking the API of
    DecoderTrainer a bit.
    """
    def __init__(self, beam_size: int) -> None:
        self._beam_size = beam_size

    def decode(self,
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
            states = next_states[:self._beam_size]
            num_steps += 1

        batch_scores = self._group_scores_by_batch(finished_states)
        loss = 0
        for scores in batch_scores.values():
            # TODO(pradeep): Minimizing the mean. Should we minimize max instead?
            loss += -torch.mean(torch.cat(scores))
        return {'loss': loss / len(batch_scores)}

    @staticmethod
    def _group_scores_by_batch(finished_states: List[DecoderState]) -> Dict[int, List[Variable]]:
        batch_scores: Dict[int, List[Variable]] = defaultdict(list)
        for state in finished_states:
            for score, batch_index in zip(state.score, state.batch_indices):
                batch_scores[batch_index].append(score)
        return batch_scores

    @classmethod
    def from_params(cls, params: Params) -> 'BeamSearchTrainer':
        beam_size = params.pop('beam_size')
        params.assert_empty(cls.__name__)
        return cls(beam_size)
