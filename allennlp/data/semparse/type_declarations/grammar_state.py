from typing import Dict, List


class GrammarState:
    """
    A ``GrammarState`` specifies the currently valid actions at every step of decoding.

    If we had a global context-free grammar, this would not be necessary - the currently valid
    actions would always be the same, and we would not need to represent the current state.
    However, our grammar is not context free (we have lambda expressions that introduce
    context-dependent production rules), and it is not global (each instance can have its own
    entities of a particular type, or its own functions).

    We thus recognize three different sources of valid actions.  The first are actions that come
    from the type declaration; these are defined once by the model and shared across all
    ``GrammarStates`` produced by that model.  The second are actions that come from the current
    instance; these are defined by the ``World`` that corresponds to each instance, and are shared
    across all decoding states for that instance.  The last are actions that come from the current
    state of the decoder; these are updated after every action taken by the decoder, though only
    some actions initiate changes.

    In practice, we use the ``World`` class to get the first two sources of valid actions at the
    same time, and we take as input a ``valid_actions`` dictionary that is computed by the
    ``World``.  These will not change during the course of decoding.  The ``GrammarState`` object
    itself maintains the context-dependent valid actions.

    Parameters
    ----------
    nonterminal_stack : ``List[str]``
        Holds the list of non-terminals that still need to be expanded.  This starts out as
        [START_SYMBOL], and decoding ends when this is empty.  Every time we take an action, we
        update the non-terminal stack and the context-dependent valid actions, and we use what's on
        the stack to decide which actions are valid in the current state.
    lambda_stack : ``Dict[str, List[str]]``
        The lambda stack keeps track of when we're in the scope of a lambda function.  The
        dictionary is keyed by the lambda variable (e.g., "x"), and the value is a nonterminal
        stack much like ``nonterminal_stack``.  When the stack becomes empty, we remove the lambda
        entry.
    valid_actions : ``Dict[str, List[int]]``
        A mapping from non-terminals (represented as strings) to all valid (global and
        instance-specific) productions from that non-terminal (represented as a list of integers).
    action_indices : ``Dict[str, int]``
        We use integers to represent productions in the ``valid_actions`` dictionary for efficiency
        reasons in the decoder.  However, we sometimes need access to the strings themselves, so we
        take this mapping from production rule strings to integers.
    """

    def __init__(self,
                 nonterminal_stack: List[str],
                 lambda_stack: Dict[str, List[str]],
                 valid_actions: Dict[str, List[int]]) -> None:
        self._nonterminal_stack = nonterminal_stack
        self._lambda_stack = lambda_stack
        self._valid_actions = valid_actions

    def is_finished(self) -> bool:
        """
        Have we finished producing our logical form?  We have finished producing the logical form
        if and only if there are no more non-terminals on the stack.
        """
        return not self._nonterminal_stack

    def get_valid_actions(self) -> List[int]:
        """
        Returns a list of valid actions (represented as integers)
        """
        return self._valid_actions[self._nonterminal_stack[-1]]

    def take_action(self, left_side: str, right_side: str) -> 'GrammarState':
        """
        Takes an action in the current grammar state, returning a new grammar state with whatever
        updates are necessary.  Because the decoder state keeps around actions that are already
        split into left hand side and right hand side, we take those here directly, instead of
        taking an action formatted as "LHS -> RHS" that we will just have to split.

        This will update the non-terminal stack and the context-dependent actions.  Updating the
        non-terminal stack involves popping the non-terminal that was expanded off of the stack,
        then pushing on any non-terminals in the production rule back on the stack.  We push the
        non-terminals on in `reverse` order, so that the first non-terminal in the production rule
        gets popped off the stack first.

        For example, if our current ``nonterminal_stack`` is ``["r", "<e,r>", "d"]``, and
        ``action`` is ``d -> [<e,d>, e]``, the resulting stack will be ``["r", "<e,r>", "e",
        "<e,d>"]``.
        """
        assert self._nonterminal_stack[-1] == left_side
        new_stack = self._nonterminal_stack[:-1]
        productions = self.get_productions_from_string(right_side)
        for production in reversed(productions):
            if self.is_nonterminal(production):
                new_stack.append(production)
        new_lambda_stack = {**self._lambda_stack}  # TODO(mattg): finish this
        return GrammarState(nonterminal_stack=new_stack,
                            lambda_stack=new_lambda_stack,
                            valid_actions=self._valid_actions)

    @staticmethod
    def get_productions_from_string(production_string: str) -> List[str]:
        """
        Takes a string like '[<d,d>, d]' and parses it into a list like ['<d,d>', 'd'].  For
        production strings that are not lists, like '<e,d>', we return a single-element list:
        ['<e,d>'].
        """
        if production_string[0] == '[':
            return production_string[1:-1].split(', ')
        else:
            return [production_string]

    @staticmethod
    def is_nonterminal(production: str) -> bool:
        # TODO(mattg): ProductionRuleField has a similar method, as does another place or two.  We
        # should centralize this logic somewhere, probably in the type declaration, or the
        # ``World`` object.
        if production[0] == '<':
            return True
        if production.startswith('fb:'):
            return False
        if len(production) > 1 or production == "x":
            return False
        return production[0].islower()
