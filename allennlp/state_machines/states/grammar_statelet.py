from typing import Callable, Dict, List, Tuple

import torch

from allennlp.nn import util


class GrammarStatelet:
    """
    A ``GrammarStatelet`` specifies the currently valid actions at every step of decoding.

    If we had a global context-free grammar, this would not be necessary - the currently valid
    actions would always be the same, and we would not need to represent the current state.
    However, our grammar is not context free (we have lambda expressions that introduce
    context-dependent production rules), and it is not global (each instance can have its own
    entities of a particular type, or its own functions).

    We thus recognize three different sources of valid actions.  The first are actions that come
    from the type declaration; these are defined once by the model and shared across all
    ``GrammarStatelets`` produced by that model.  The second are actions that come from the current
    instance; these are defined by the ``World`` that corresponds to each instance, and are shared
    across all decoding states for that instance.  The last are actions that come from the current
    state of the decoder; these are updated after every action taken by the decoder, though only
    some actions initiate changes.

    In practice, we use the ``World`` class to get the first two sources of valid actions at the
    same time, and we take as input a ``valid_actions`` dictionary that is computed by the
    ``World``.  These will not change during the course of decoding.  The ``GrammarStatelet``
    object itself maintains the context-dependent valid actions.

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
    valid_actions : ``Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]]``
        A mapping from non-terminals (represented as strings) to all valid expansions of that
        non-terminal.  The way we represent the valid expansions is a little complicated: we use a
        dictionary of `action types`, where the key is the action type (like "global", "linked", or
        whatever your model is expecting), and the value is a tuple representing all actions of
        that type.  The tuple is (input tensor, output tensor, action id).  The input tensor has
        the representation that is used when `selecting` actions, for all actions of this type.
        The output tensor has the representation that is used when feeding the action to the next
        step of the decoder (this could just be the same as the input tensor).  The action ids are
        a list of indices into the main action list for each batch instance.
    context_actions : ``Dict[str, Tuple[torch.Tensor, torch.Tensor, int]]``
        Variable actions are never included in the ``valid_actions`` dictionary, because they are
        only valid depending on the current grammar state.  This dictionary maps from the string
        representation of all such actions to the tensor representations of the actions.  These
        will get added onto the "global" key in the ``valid_actions`` when they are allowed.
    is_nonterminal : ``Callable[[str], bool]``
        A function that is used to determine whether each piece of the RHS of the action string is
        a non-terminal that needs to be added to the non-terminal stack.  You can use
        ``type_declaraction.is_nonterminal`` here, or write your own function if that one doesn't
        work for your domain.
    """
    def __init__(self,
                 nonterminal_stack: List[str],
                 lambda_stacks: Dict[Tuple[str, str], List[str]],
                 valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]],
                 context_actions: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]],
                 is_nonterminal: Callable[[str], bool]) -> None:
        self._nonterminal_stack = nonterminal_stack
        self._lambda_stacks = lambda_stacks
        self._valid_actions = valid_actions
        self._context_actions = context_actions
        self._is_nonterminal = is_nonterminal

    def is_finished(self) -> bool:
        """
        Have we finished producing our logical form?  We have finished producing the logical form
        if and only if there are no more non-terminals on the stack.
        """
        return not self._nonterminal_stack

    def get_valid_actions(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Returns the valid actions in the current grammar state.  See the class docstring for a
        description of what we're returning here.
        """
        actions = self._valid_actions[self._nonterminal_stack[-1]]
        context_actions = []
        for type_, variable in self._lambda_stacks:
            if self._nonterminal_stack[-1] == type_:
                production_string = f"{type_} -> {variable}"
                context_actions.append(self._context_actions[production_string])
        if context_actions:
            input_tensor, output_tensor, action_ids = actions['global']
            new_inputs = [input_tensor] + [x[0] for x in context_actions]
            input_tensor = torch.cat(new_inputs, dim=0)
            new_outputs = [output_tensor] + [x[1] for x in context_actions]
            output_tensor = torch.cat(new_outputs, dim=0)
            new_action_ids = action_ids + [x[2] for x in context_actions]
            # We can't just reassign to actions['global'], because that would modify the state of
            # self._valid_actions.  Instead, we need to construct a new actions dictionary.
            new_actions = {**actions}
            new_actions['global'] = (input_tensor, output_tensor, new_action_ids)
            actions = new_actions
        return actions

    def take_action(self, production_rule: str) -> 'GrammarStatelet':
        """
        Takes an action in the current grammar state, returning a new grammar state with whatever
        updates are necessary.  The production rule is assumed to be formatted as "LHS -> RHS".

        This will update the non-terminal stack and the context-dependent actions.  Updating the
        non-terminal stack involves popping the non-terminal that was expanded off of the stack,
        then pushing on any non-terminals in the production rule back on the stack.  We push the
        non-terminals on in `reverse` order, so that the first non-terminal in the production rule
        gets popped off the stack first.

        For example, if our current ``nonterminal_stack`` is ``["r", "<e,r>", "d"]``, and
        ``action`` is ``d -> [<e,d>, e]``, the resulting stack will be ``["r", "<e,r>", "e",
        "<e,d>"]``.
        """
        left_side, right_side = production_rule.split(' -> ')
        assert self._nonterminal_stack[-1] == left_side, (f"Tried to expand {self._nonterminal_stack[-1]}"
                                                          f"but got rule {left_side} -> {right_side}")
        assert all(self._lambda_stacks[key][-1] == left_side for key in self._lambda_stacks)

        new_stack = self._nonterminal_stack[:-1]
        new_lambda_stacks = {key: self._lambda_stacks[key][:-1] for key in self._lambda_stacks}

        productions = self._get_productions_from_string(right_side)
        # Looking for lambda productions, but not for cells or columns with the word "lambda" in
        # them.
        if 'lambda' in productions[0] and 'fb:' not in productions[0]:
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
        new_lambda_stacks = {key: new_lambda_stacks[key]
                             for key in new_lambda_stacks if new_lambda_stacks[key]}

        return GrammarStatelet(nonterminal_stack=new_stack,
                               lambda_stacks=new_lambda_stacks,
                               valid_actions=self._valid_actions,
                               context_actions=self._context_actions,
                               is_nonterminal=self._is_nonterminal)

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
                    self._lambda_stacks == other._lambda_stacks,
                    util.tensors_equal(self._valid_actions, other._valid_actions),
                    util.tensors_equal(self._context_actions, other._context_actions),
                    self._is_nonterminal == other._is_nonterminal,
                    ])
        return NotImplemented
