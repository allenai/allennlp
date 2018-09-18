from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import torch

from allennlp.state_machines.states.grammar_based_state import GrammarBasedState 

def is_nonterminal(token: str):
    if token[0] == '"' and token[-1] == '"':
        return False
    return True


class AtisGrammarBasedState(GrammarBasedState):
    def get_valid_actions(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Returns the valid actions in the current grammar state.  See the class docstring for a
        description of what we're returning here.
        """
        actions = self._valid_actions[self._nonterminal_stack[-1]]

        '''
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
            actions['global'] = (input_tensor, output_tensor, new_action_ids)
        '''

        return actions


    def take_action(self, production_rule: str) -> 'GrammarState':
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

            for production in productions:
                if self._is_nonterminal(production):
                    new_stack.append(production)
                    for lambda_stack in new_lambda_stacks.values():
                        lambda_stack.append(production)

            # If any of the lambda stacks have now become empty, we remove them from our dictionary.
            new_lambda_stacks = {key: new_lambda_stacks[key]
                                 for key in new_lambda_stacks if new_lambda_stacks[key]}

            return AtisGrammarState(nonterminal_stack=new_stack,
                                lambda_stacks=new_lambda_stacks,
                                valid_actions=self._valid_actions,
                                context_actions=self._context_actions,
                                is_nonterminal=self._is_nonterminal)
