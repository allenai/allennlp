from typing import Callable, Dict, Generic, List, TypeVar

from allennlp.nn import util

ActionRepresentation = TypeVar('ActionRepresentation')  # pylint: disable=invalid-name


class GrammarStatelet(Generic[ActionRepresentation]):
    """
    A ``GrammarStatelet`` keeps track of the currently valid actions at every step of decoding.

    This class is relatively simple: we have a non-terminal stack which tracks which non-terminals
    we still need to expand.  At every timestep of decoding, we take an action that pops something
    off of the non-terminal stack, and possibly pushes more things on.  The grammar state is
    "finished" when the non-terminal stack is empty.

    At any point during decoding, you can query this object to get a representation of all of the
    valid actions in the current state.  The representation is something that you provide when
    constructing the initial state, in whatever form you want, and we just hold on to it for you
    and return it when you ask.  Putting this in here is purely for convenience, to group together
    pieces of state that are related to taking actions - if you want to handle the action
    representations outside of this class, that would work just fine too.

    Parameters
    ----------
    nonterminal_stack : ``List[str]``
        Holds the list of non-terminals that still need to be expanded.  This starts out as
        [START_SYMBOL], and decoding ends when this is empty.  Every time we take an action, we
        update the non-terminal stack and the context-dependent valid actions, and we use what's on
        the stack to decide which actions are valid in the current state.
    valid_actions : ``Dict[str, ActionRepresentation]``
        A mapping from non-terminals (represented as strings) to all valid expansions of that
        non-terminal.  The class that constructs this object can pick how it wants the actions to
        be represented.
    is_nonterminal : ``Callable[[str], bool]``
        A function that is used to determine whether each piece of the RHS of the action string is
        a non-terminal that needs to be added to the non-terminal stack.  You can use
        ``type_declaraction.is_nonterminal`` here, or write your own function if that one doesn't
        work for your domain.
    reverse_productions: ``bool``, optional (default=True)
        A flag that reverses the production rules when ``True``. If the production rules are
        reversed, then the first non-terminal in the production will be popped off the stack first,
        giving us left-to-right production.  If this is ``False``, you will get right-to-left
        production.
    """
    def __init__(self,
                 nonterminal_stack: List[str],
                 valid_actions: Dict[str, ActionRepresentation],
                 is_nonterminal: Callable[[str], bool],
                 reverse_productions: bool = True) -> None:
        self._nonterminal_stack = nonterminal_stack
        self._valid_actions = valid_actions
        self._is_nonterminal = is_nonterminal
        self._reverse_productions = reverse_productions

    def is_finished(self) -> bool:
        """
        Have we finished producing our logical form?  We have finished producing the logical form
        if and only if there are no more non-terminals on the stack.
        """
        return not self._nonterminal_stack

    def get_valid_actions(self) -> ActionRepresentation:
        """
        Returns the valid actions in the current grammar state.  The `Model` determines what
        exactly this looks like when it constructs the `valid_actions` dictionary.
        """
        return self._valid_actions[self._nonterminal_stack[-1]]

    def take_action(self, production_rule: str) -> 'GrammarStatelet':
        """
        Takes an action in the current grammar state, returning a new grammar state with whatever
        updates are necessary.  The production rule is assumed to be formatted as "LHS -> RHS".

        This will update the non-terminal stack.  Updating the non-terminal stack involves popping
        the non-terminal that was expanded off of the stack, then pushing on any non-terminals in
        the production rule back on the stack.

        For example, if our current ``nonterminal_stack`` is ``["r", "<e,r>", "d"]``, and
        ``action`` is ``d -> [<e,d>, e]``, the resulting stack will be ``["r", "<e,r>", "e",
        "<e,d>"]``.

        If ``self._reverse_productions`` is set to ``False`` then we push the non-terminals on in
        in their given order, which means that the first non-terminal in the production rule gets
        popped off the stack `last`.
        """
        left_side, right_side = production_rule.split(' -> ')
        assert self._nonterminal_stack[-1] == left_side, (f"Tried to expand {self._nonterminal_stack[-1]}"
                                                          f"but got rule {left_side} -> {right_side}")

        new_stack = self._nonterminal_stack[:-1]

        productions = self._get_productions_from_string(right_side)
        if self._reverse_productions:
            productions = list(reversed(productions))

        for production in productions:
            if self._is_nonterminal(production):
                new_stack.append(production)

        return GrammarStatelet(nonterminal_stack=new_stack,
                               valid_actions=self._valid_actions,
                               is_nonterminal=self._is_nonterminal,
                               reverse_productions=self._reverse_productions)

    @staticmethod
    def _get_productions_from_string(production_string: str) -> List[str]:
        """
        Takes a string like '[<d,d>, d]' and parses it into a list like ['<d,d>', 'd'].  For
        production strings that are not lists, like '<e,d>', we return a single-element list:
        ['<e,d>'].
        """
        if production_string[0] == '[':
            return production_string[1:-1].split(', ')
        else:
            return [production_string]

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            # pylint: disable=protected-access
            return all([
                    self._nonterminal_stack == other._nonterminal_stack,
                    util.tensors_equal(self._valid_actions, other._valid_actions),
                    self._is_nonterminal == other._is_nonterminal,
                    self._reverse_productions == other._reverse_productions,
                    ])
        return NotImplemented
