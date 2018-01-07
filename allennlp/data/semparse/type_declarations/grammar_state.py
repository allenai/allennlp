from copy import deepcopy
from typing import Dict, List, Tuple


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
    lambda_stacks : ``Dict[Tuple[str, str], List[str]]``
        The lambda stack keeps track of when we're in the scope of a lambda function.  The
        dictionary is keyed by the production rule we are adding (like "r -> x", separated into
        left hand side and right hand side, where the LHS is the type of the lambda variable and
        the RHS is the variable itself), and the value is a nonterminal stack much like
        ``nonterminal_stack``.  When the stack becomes empty, we remove the lambda entry.
    valid_actions : ``Dict[str, List[int]]``
        A mapping from non-terminals (represented as strings) to all valid (global and
        instance-specific) productions from that non-terminal (represented as a list of integers).
    action_indices : ``Dict[str, int]``
        We use integers to represent productions in the ``valid_actions`` dictionary for efficiency
        reasons in the decoder.  This means we need a way to map from the production rule strings
        that we generate for lambda variables back to the integer used to represent it.
    """
    # TODO(mattg): We're hard-coding three lambda variables here.  This isn't a great way to do
    # this; it's just something that works for now, that we can fix later if / when it's needed.
    # If you allow for more than three nested lambdas, or if you want to use different lambda
    # variable names, you'll have to change this somehow.
    lambda_variables = set(['x', 'y', 'z'])

    def __init__(self,
                 nonterminal_stack: List[str],
                 lambda_stacks: Dict[Tuple[str, str], List[str]],
                 valid_actions: Dict[str, List[int]],
                 action_indices: Dict[str, int]) -> None:
        self._nonterminal_stack = nonterminal_stack
        self._lambda_stacks = lambda_stacks
        self._valid_actions = valid_actions
        self._action_indices = action_indices

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
        actions = self._valid_actions[self._nonterminal_stack[-1]]
        for type_, variable in self._lambda_stacks:
            if self._nonterminal_stack[-1] == type_:
                production_string = f"{type_} -> {variable}"
                actions = actions + [self._action_indices[production_string]]
        return actions

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
        new_lambda_stacks = deepcopy(self._lambda_stacks)
        for key, lambda_stack in new_lambda_stacks.items():
            assert lambda_stack[-1] == left_side
            lambda_stack.pop()  # pop to modify the value in the dictionary

        productions = self._get_productions_from_string(right_side)
        if 'lambda' in productions[0]:
            production = productions[0]
            if production[0] == "'" and production[-1] == "'":
                # The production rule with a lambda is typically "<t,d> -> ['lambda x', d]".  We
                # need to strip the quotes.
                production = production[1:-1]
            lambda_variable = production.split(' ')[1]
            # The left side must be formatted as "<t,d>", where "t" is the type of the lambda
            # variable, and "d" is the return type of the lambda function.  We need to pull out the
            # "t" here.  TODO(mattg): this is pretty limiting, but I'm not sure how general we
            # should make this.
            if len(left_side) != 5:
                raise NotImplementedError("Can't handle this type yet:", left_side)
            lambda_type = left_side[1]
            new_lambda_stacks[(lambda_type, lambda_variable)] = []

        for production in reversed(productions):
            if self._is_nonterminal(production):
                new_stack.append(production)
                for lambda_stack in new_lambda_stacks.values():
                    lambda_stack.append(production)

        # If any of the lambda stacks have now become empty, we remove them from our dictionary.
        finished_lambdas = set()
        for key, lambda_stack in new_lambda_stacks.items():
            if not lambda_stack:
                finished_lambdas.add(key)
        for finished_lambda in finished_lambdas:
            del new_lambda_stacks[finished_lambda]

        return GrammarState(nonterminal_stack=new_stack,
                            lambda_stacks=new_lambda_stacks,
                            valid_actions=self._valid_actions,
                            action_indices=self._action_indices)

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

    @staticmethod
    def _is_nonterminal(production: str) -> bool:
        # TODO(mattg): ProductionRuleField has a similar method, as does another place or two.  We
        # should centralize this logic somewhere, probably in the type declaration, or the
        # ``World`` object.  This is pretty specific to the WikiTablesWorld, and to the assumptions
        # made in converting types to strings (e.g., that we're only using the first letter for
        # types, lowercased).
        if production in ['<=', '<']:
            # Some grammars (including the wikitables grammar) have "less than" and "less than or
            # equal to" functions that are terminals.  We don't want to treat those like our
            # "<t,d>" types.
            return False
        if production[0] == '<':
            return True
        if production.startswith('fb:'):
            return False
        if len(production) > 1 or production in GrammarState.lambda_variables:
            return False
        return production[0].islower()
