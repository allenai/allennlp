# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.decoding import GrammarState

def is_nonterminal(symbol: str) -> bool:
    if symbol == 'identity':
        return False
    if 'lambda ' in symbol:
        return False
    if symbol in {'x', 'y', 'z'}:
        return False
    return True


class TestGrammarState(AllenNlpTestCase):
    def test_is_finished_just_uses_nonterminal_stack(self):
        state = GrammarState(['s'], {}, {}, {}, is_nonterminal)
        assert not state.is_finished()
        state = GrammarState([], {}, {}, {}, is_nonterminal)
        assert state.is_finished()

    def test_get_valid_actions_uses_top_of_stack(self):
        state = GrammarState(['s'], {}, {'s': [1, 2], 't': [3, 4]}, {}, is_nonterminal)
        assert state.get_valid_actions() == [1, 2]
        state = GrammarState(['t'], {}, {'s': [1, 2], 't': [3, 4]}, {}, is_nonterminal)
        assert state.get_valid_actions() == [3, 4]
        state = GrammarState(['e'], {}, {'s': [1, 2], 't': [3, 4], 'e': []}, {}, is_nonterminal)
        assert state.get_valid_actions() == []

    def test_get_valid_actions_adds_lambda_productions(self):
        state = GrammarState(['s'], {('s', 'x'): ['s']}, {'s': [1, 2]}, {'s -> x': 5}, is_nonterminal)
        assert state.get_valid_actions() == [1, 2, 5]
        # We're doing this assert twice to make sure we haven't accidentally modified the state.
        assert state.get_valid_actions() == [1, 2, 5]

    def test_get_valid_actions_adds_lambda_productions_only_for_correct_type(self):
        state = GrammarState(['t'],
                             {('s', 'x'): ['t']},
                             {'s': [1, 2], 't': [3, 4]},
                             {'s -> x': 5},
                             is_nonterminal)
        assert state.get_valid_actions() == [3, 4]
        # We're doing this assert twice to make sure we haven't accidentally modified the state.
        assert state.get_valid_actions() == [3, 4]

    def test_take_action_gives_correct_next_states_with_non_lambda_productions(self):
        # state.take_action() doesn't read or change these objects, it just passes them through, so
        # we'll use some sentinels to be sure of that.
        valid_actions = object()
        action_indices = object()

        state = GrammarState(['s'], {}, valid_actions, action_indices, is_nonterminal)
        next_state = state.take_action('s -> [t, r]')
        expected_next_state = GrammarState(['r', 't'], {}, valid_actions, action_indices, is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = GrammarState(['r', 't'], {}, valid_actions, action_indices, is_nonterminal)
        next_state = state.take_action('t -> identity')
        expected_next_state = GrammarState(['r'], {}, valid_actions, action_indices, is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

    def test_take_action_crashes_with_mismatched_types(self):
        with pytest.raises(AssertionError):
            state = GrammarState(['s'], {}, {}, {}, is_nonterminal)
            state.take_action('t -> identity')

    def test_take_action_gives_correct_next_states_with_lambda_productions(self):
        # state.take_action() doesn't read or change these objects, it just passes them through, so
        # we'll use some sentinels to be sure of that.
        valid_actions = object()
        action_indices = object()

        state = GrammarState(['t', '<s,d>'], {}, valid_actions, action_indices, is_nonterminal)
        next_state = state.take_action('<s,d> -> [lambda x, d]')
        expected_next_state = GrammarState(['t', 'd'],
                                           {('s', 'x'): ['d']},
                                           valid_actions,
                                           action_indices,
                                           is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action('d -> [<s,r>, d]')
        expected_next_state = GrammarState(['t', 'd', '<s,r>'],
                                           {('s', 'x'): ['d', '<s,r>']},
                                           valid_actions,
                                           action_indices,
                                           is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action('<s,r> -> [lambda y, r]')
        expected_next_state = GrammarState(['t', 'd', 'r'],
                                           {('s', 'x'): ['d', 'r'], ('s', 'y'): ['r']},
                                           valid_actions,
                                           action_indices,
                                           is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action('r -> identity')
        expected_next_state = GrammarState(['t', 'd'],
                                           {('s', 'x'): ['d']},
                                           valid_actions,
                                           action_indices,
                                           is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action('d -> x')
        expected_next_state = GrammarState(['t'],
                                           {},
                                           valid_actions,
                                           action_indices,
                                           is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__
