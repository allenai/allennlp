# pylint: disable=no-self-use,invalid-name
import pytest
import torch
from numpy.testing import assert_almost_equal

from allennlp.common.testing import AllenNlpTestCase
from allennlp.state_machines.states import LambdaGrammarStatelet

def is_nonterminal(symbol: str) -> bool:
    if symbol == 'identity':
        return False
    if 'lambda ' in symbol:
        return False
    if symbol in {'x', 'y', 'z'}:
        return False
    return True


class TestLambdaGrammarStatelet(AllenNlpTestCase):
    def test_is_finished_just_uses_nonterminal_stack(self):
        state = LambdaGrammarStatelet(['s'], {}, {}, {}, is_nonterminal)
        assert not state.is_finished()
        state = LambdaGrammarStatelet([], {}, {}, {}, is_nonterminal)
        assert state.is_finished()

    def test_get_valid_actions_uses_top_of_stack(self):
        s_actions = object()
        t_actions = object()
        e_actions = object()
        state = LambdaGrammarStatelet(['s'], {}, {'s': s_actions, 't': t_actions}, {}, is_nonterminal)
        assert state.get_valid_actions() == s_actions
        state = LambdaGrammarStatelet(['t'], {}, {'s': s_actions, 't': t_actions}, {}, is_nonterminal)
        assert state.get_valid_actions() == t_actions
        state = LambdaGrammarStatelet(['e'],
                                      {},
                                      {'s': s_actions, 't': t_actions, 'e': e_actions},
                                      {},
                                      is_nonterminal)
        assert state.get_valid_actions() == e_actions

    def test_get_valid_actions_adds_lambda_productions(self):
        state = LambdaGrammarStatelet(['s'],
                                      {('s', 'x'): ['s']},
                                      {'s': {'global': (torch.Tensor([1, 1]), torch.Tensor([2, 2]), [1, 2])}},
                                      {'s -> x': (torch.Tensor([5]), torch.Tensor([6]), 5)},
                                      is_nonterminal)
        actions = state.get_valid_actions()
        assert_almost_equal(actions['global'][0].cpu().numpy(), [1, 1, 5])
        assert_almost_equal(actions['global'][1].cpu().numpy(), [2, 2, 6])
        assert actions['global'][2] == [1, 2, 5]
        # We're doing this assert twice to make sure we haven't accidentally modified the state.
        actions = state.get_valid_actions()
        assert_almost_equal(actions['global'][0].cpu().numpy(), [1, 1, 5])
        assert_almost_equal(actions['global'][1].cpu().numpy(), [2, 2, 6])
        assert actions['global'][2] == [1, 2, 5]

    def test_get_valid_actions_adds_lambda_productions_only_for_correct_type(self):
        state = LambdaGrammarStatelet(['t'],
                                      {('s', 'x'): ['t']},
                                      {'s': {'global': (torch.Tensor([1, 1]), torch.Tensor([2, 2]), [1, 2])},
                                       't': {'global': (torch.Tensor([3, 3]), torch.Tensor([4, 4]), [3, 4])}},
                                      {'s -> x': (torch.Tensor([5]), torch.Tensor([6]), 5)},
                                      is_nonterminal)
        actions = state.get_valid_actions()
        assert_almost_equal(actions['global'][0].cpu().numpy(), [3, 3])
        assert_almost_equal(actions['global'][1].cpu().numpy(), [4, 4])
        assert actions['global'][2] == [3, 4]
        # We're doing this assert twice to make sure we haven't accidentally modified the state.
        actions = state.get_valid_actions()
        assert_almost_equal(actions['global'][0].cpu().numpy(), [3, 3])
        assert_almost_equal(actions['global'][1].cpu().numpy(), [4, 4])
        assert actions['global'][2] == [3, 4]

    def test_take_action_gives_correct_next_states_with_non_lambda_productions(self):
        # state.take_action() doesn't read or change these objects, it just passes them through, so
        # we'll use some sentinels to be sure of that.
        valid_actions = object()
        context_actions = object()

        state = LambdaGrammarStatelet(['s'], {}, valid_actions, context_actions, is_nonterminal)
        next_state = state.take_action('s -> [t, r]')
        expected_next_state = LambdaGrammarStatelet(['r', 't'], {}, valid_actions, context_actions, is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = LambdaGrammarStatelet(['r', 't'], {}, valid_actions, context_actions, is_nonterminal)
        next_state = state.take_action('t -> identity')
        expected_next_state = LambdaGrammarStatelet(['r'], {}, valid_actions, context_actions, is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

    def test_take_action_crashes_with_mismatched_types(self):
        with pytest.raises(AssertionError):
            state = LambdaGrammarStatelet(['s'], {}, {}, {}, is_nonterminal)
            state.take_action('t -> identity')

    def test_take_action_gives_correct_next_states_with_lambda_productions(self):
        # state.take_action() doesn't read or change these objects, it just passes them through, so
        # we'll use some sentinels to be sure of that.
        valid_actions = object()
        context_actions = object()

        state = LambdaGrammarStatelet(['t', '<s,d>'], {}, valid_actions, context_actions, is_nonterminal)
        next_state = state.take_action('<s,d> -> [lambda x, d]')
        expected_next_state = LambdaGrammarStatelet(['t', 'd'],
                                                    {('s', 'x'): ['d']},
                                                    valid_actions,
                                                    context_actions,
                                                    is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action('d -> [<s,r>, d]')
        expected_next_state = LambdaGrammarStatelet(['t', 'd', '<s,r>'],
                                                    {('s', 'x'): ['d', '<s,r>']},
                                                    valid_actions,
                                                    context_actions,
                                                    is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action('<s,r> -> [lambda y, r]')
        expected_next_state = LambdaGrammarStatelet(['t', 'd', 'r'],
                                                    {('s', 'x'): ['d', 'r'], ('s', 'y'): ['r']},
                                                    valid_actions,
                                                    context_actions,
                                                    is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action('r -> identity')
        expected_next_state = LambdaGrammarStatelet(['t', 'd'],
                                                    {('s', 'x'): ['d']},
                                                    valid_actions,
                                                    context_actions,
                                                    is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action('d -> x')
        expected_next_state = LambdaGrammarStatelet(['t'],
                                                    {},
                                                    valid_actions,
                                                    context_actions,
                                                    is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__
