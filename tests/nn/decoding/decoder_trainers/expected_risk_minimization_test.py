"""
We define a simple deterministic decoder here, that takes steps to add integers to list. At
each step, the decoder takes the last integer in the list, and adds either 1 or 2 to produce the
next element that will be added to the list. We initialize the list to contain one element, 0 and
we say that a sequence is finished when the last element is 4. We define the score of a state as the
negative of the number of elements (excluding 0) in the action history, and the cost of a finished
state as the number of odd numbers in the list.
"""


# pylint: disable=no-self-use,protected-access
from typing import List, Set, Dict
from collections import defaultdict

from overrides import overrides
import torch
from torch.autograd import Variable
import numpy as np
from numpy.testing import assert_almost_equal

from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_trainers import ExpectedRiskMinimization


class SimpleDecoderState(DecoderState['SimpleDecoderState']):
    def is_finished(self) -> bool:
        return self.action_history[0][-1] == 4

    @classmethod
    def combine_states(cls, states) -> 'SimpleDecoderState':
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in
                            state.action_history]
        scores = [score for state in states for score in state.score]
        return SimpleDecoderState(batch_indices, action_histories, scores)


class SimpleDecoderStep(DecoderStep[SimpleDecoderState]):
    @overrides
    def take_step(self,
                  state: SimpleDecoderState,
                  max_actions: int = None,
                  allowed_actions: List[Set] = None) -> List[SimpleDecoderState]:
        if allowed_actions is None:
            # For each element in the group, the allowed actions are adding 1 or 2 to the last
            # element.
            allowed_actions = [{1, 2} for _ in state.batch_indices]
        indexed_next_states: Dict[int, List[SimpleDecoderState]] = defaultdict(list)
        for batch_index, action_history, score, actions in zip(state.batch_indices,
                                                               state.action_history,
                                                               state.score,
                                                               allowed_actions):

            for action in actions:
                next_item = action_history[-1] + action
                new_history = action_history + [next_item]
                # For every action taken, we reduce the score by 1.
                new_score = score - 1
                new_state = SimpleDecoderState([batch_index],
                                               [new_history],
                                               [new_score])
                indexed_next_states[batch_index].append(new_state)
        next_states: List[SimpleDecoderState] = []
        for batch_next_states in indexed_next_states.values():
            if max_actions is not None:
                batch_next_states = batch_next_states[:max_actions]
            next_states.extend(batch_next_states)
        return next_states


class TestExpectedRiskMinimization(AllenNlpTestCase):
    def setUp(self):
        super(TestExpectedRiskMinimization, self).setUp()
        self.initial_state = SimpleDecoderState([0], [[0]], [Variable(torch.Tensor([0.0]))])
        self.decoder_step = SimpleDecoderStep()
        # Cost is the number of odd elements in the action history.
        self.supervision = lambda state: Variable(torch.Tensor([sum([x%2 != 0 for x in
                                                                     state.action_history[0]])]))
        # High beam size ensures exhaustive search.
        self.trainer = ExpectedRiskMinimization(beam_size=100,
                                                normalize_by_length=False,
                                                max_decoding_steps=10)

    def test_get_finished_states(self):
        finished_states = self.trainer._get_finished_states(self.initial_state, self.decoder_step)
        state_info = [(state.action_history[0], int(state.score[0].data)) for state in finished_states]
        # There will be exactly five finished states with the following paths. Each score is the
        # negative of one less than the number of elements in the action history.
        assert len(finished_states) == 5
        assert ([0, 2, 4], -2) in state_info
        assert ([0, 1, 2, 4], -3) in state_info
        assert ([0, 1, 3, 4], -3) in state_info
        assert ([0, 2, 3, 4], -3) in state_info
        assert ([0, 1, 2, 3, 4], -4) in state_info

    def test_decode(self):
        decoded_info = self.trainer.decode(self.initial_state, self.decoder_step, self.supervision)
        # The best state corresponds to the shortest path.
        assert decoded_info['best_action_sequence'][0] == [0, 2, 4]
        # The scores and costs corresponding to the finished states will be
        # [0, 2, 4] : -2, 0
        # [0, 1, 2, 4] : -3, 1
        # [0, 1, 3, 4] : -3, 2
        # [0, 2, 3, 4] : -3, 1
        # [0, 1, 2, 3, 4] : -4, 2

        # This is the normalization factor while re-normalizing probabilities on the beam
        partition = np.exp(-2) + np.exp(-3) + np.exp(-3) + np.exp(-3) + np.exp(-4)
        expected_loss = ((np.exp(-2) * 0) + (np.exp(-3) * 1) + (np.exp(-3) * 2) +
                         (np.exp(-3) *1) + (np.exp(-4) * 2)) / partition
        assert_almost_equal(decoded_info['loss'].data.numpy(), expected_loss)
