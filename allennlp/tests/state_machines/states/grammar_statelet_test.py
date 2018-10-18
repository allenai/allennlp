# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.state_machines.states import GrammarStatelet

def is_nonterminal(symbol: str) -> bool:
    if symbol == 'identity':
        return False
    if 'lambda ' in symbol:
        return False
    if symbol in {'x', 'y', 'z'}:
        return False
    return True


class TestGrammarStatelet(AllenNlpTestCase):
    def test_is_finished_just_uses_nonterminal_stack(self):
        state = GrammarStatelet(['s'], {}, is_nonterminal)
        assert not state.is_finished()
        state = GrammarStatelet([], {}, is_nonterminal)
        assert state.is_finished()

    def test_get_valid_actions_uses_top_of_stack(self):
        s_actions = object()
        t_actions = object()
        e_actions = object()
        state = GrammarStatelet(['s'], {'s': s_actions, 't': t_actions}, is_nonterminal)
        assert state.get_valid_actions() == s_actions
        state = GrammarStatelet(['t'], {'s': s_actions, 't': t_actions}, is_nonterminal)
        assert state.get_valid_actions() == t_actions
        state = GrammarStatelet(['e'], {'s': s_actions, 't': t_actions, 'e': e_actions}, is_nonterminal)
        assert state.get_valid_actions() == e_actions

    def test_take_action_crashes_with_mismatched_types(self):
        with pytest.raises(AssertionError):
            state = GrammarStatelet(['s'], {}, is_nonterminal)
            state.take_action('t -> identity')
