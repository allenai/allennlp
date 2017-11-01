from typing import List

from allennlp.common import Params
from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState


class BeamSearch:
    """
    This class implements beam search over transition sequences given an initial ``DecoderState``
    and a ``DecoderStep``, returning the highest scoring final states found by the beam (the states
    will keep track of the transition sequence themselves).
    """
    def __init__(self, beam_size: int) -> None:
        self._beam_size = beam_size

    def search(self,
               num_steps: int,
               initial_state: DecoderState,
               decoder_step: DecoderStep,
               keep_final_unfinished_states: bool = True) -> List[DecoderState]:
        finished_states = []
        states = [initial_state]
        step_num = 1
        while states and step_num <= num_steps:
            step_num += 1
            next_states = []
            for state in states:
                next_state_generator = decoder_step.take_step(state)
                for _ in range(self._beam_size):
                    next_state = next(next_state_generator)
                    if next_state.is_finished():
                        finished_states.append(next_state)
                    else:
                        if step_num == num_steps and keep_final_unfinished_states:
                            finished_states.append(next_state)
                        next_states.append(next_state)
            to_sort = [(-state.score.data[0], state) for state in next_states]
            to_sort.sort(key=lambda x: x[0])
            states = [state[1] for state in to_sort[:self._beam_size]]
        finished_to_sort = [(-state.score.data[0], state) for state in finished_states]
        finished_to_sort.sort(key=lambda x: x[0])
        return [state[1] for state in finished_to_sort]

    @classmethod
    def from_params(cls, params: Params) -> 'BeamSearch':
        beam_size = params.pop('beam_size')
        return cls(beam_size=beam_size)
