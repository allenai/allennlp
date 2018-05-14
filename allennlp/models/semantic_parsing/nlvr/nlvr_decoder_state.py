from typing import List, Dict, Tuple

import torch
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.nn.decoding import DecoderState, GrammarState, RnnState, ChecklistState
from allennlp.semparse.worlds import NlvrWorld


class NlvrDecoderState(DecoderState['NlvrDecoderState']):
    """
    This class is very similar to ``WikiTablesDecoderState``, except that we keep track of a
    checklist score, and other variables related to it.

    Parameters
    ----------
    batch_indices : ``List[int]``
        Passed to super class; see docs there.
    action_history : ``List[List[int]]``
        Passed to super class; see docs there.
    score : ``List[torch.Tensor]``
        Passed to super class; see docs there.
    rnn_state : ``List[RnnState]``
        An ``RnnState`` for every group element.  This keeps track of the current decoder hidden
        state, the previous decoder output, the output from the encoder (for computing attentions),
        and other things that are typical seq2seq decoder state things.
    grammar_state : ``List[GrammarState]``
        This hold the current grammar state for each element of the group.  The ``GrammarState``
        keeps track of which actions are currently valid.
    action_embeddings : ``torch.Tensor``
        The global action embeddings tensor.  Has shape ``(num_global_embeddable_actions,
        action_embedding_dim)``.
    action_indices : ``Dict[Tuple[int, int], int]``
        A mapping from ``(batch_index, action_index)`` to ``global_action_index``.
    possible_actions : ``List[List[ProductionRuleArray]]``
        The list of all possible actions that was passed to ``model.forward()``.  We need this so
        we can recover production strings, which we need to update grammar states.
    worlds : ``List[List[NlvrWorld]]``
        The worlds associated with each element. These are needed to compute the denotations. The
        outer list corresponds to elements, and the inner corresponds to worlds related to each
        element.
    label_strings : ``List[List[str]]``
        String representations of labels for the elements provided. When scoring finished states, we
        will compare the denotations of their action sequences against these labels. For each
        element, there will be as many labels as there are worlds.
    checklist_state : ``List[ChecklistState]``, optional (default=None)
        If you are using this state within a parser being trained for coverage, we need to store a
        ``ChecklistState`` which keeps track of the coverage information. Not needed if you are
        using a non-coverage based training algorithm.
    """
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor],
                 rnn_state: List[RnnState],
                 grammar_state: List[GrammarState],
                 action_embeddings: torch.Tensor,
                 action_indices: Dict[Tuple[int, int], int],
                 possible_actions: List[List[ProductionRuleArray]],
                 worlds: List[List[NlvrWorld]],
                 label_strings: List[List[str]],
                 checklist_state: List[ChecklistState] = None) -> None:
        super(NlvrDecoderState, self).__init__(batch_indices, action_history, score)
        self.rnn_state = rnn_state
        self.grammar_state = grammar_state
        # Converting None to list of Nones if needed, to simplify state operations.
        self.checklist_state = checklist_state if checklist_state is not None else [None for _ in
                                                                                    batch_indices]
        self.action_embeddings = action_embeddings
        self.action_indices = action_indices
        self.possible_actions = possible_actions
        self.worlds = worlds
        self.label_strings = label_strings

    def print_action_history(self, group_index: int = None) -> None:
        scores = self.score if group_index is None else [self.score[group_index]]
        batch_indices = self.batch_indices if group_index is None else [self.batch_indices[group_index]]
        histories = self.action_history if group_index is None else [self.action_history[group_index]]
        for score, batch_index, action_history in zip(scores, batch_indices, histories):
            print('  ', score.data.cpu().numpy()[0],
                  [self.possible_actions[batch_index][action][0] for action in action_history])

    def get_valid_actions(self) -> List[List[int]]:
        """
        Returns a list of valid actions for each element of the group.
        """
        valid_actions = [state.get_valid_actions() for state in self.grammar_state]
        return valid_actions

    def is_finished(self) -> bool:
        """This method is identical to ``WikiTablesDecoderState.is_finished``."""
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_state[0].is_finished()

    @classmethod
    def combine_states(cls, states) -> 'NlvrDecoderState':
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        rnn_states = [rnn_state for state in states for rnn_state in state.rnn_state]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        checklist_states = [checklist_state for state in states for checklist_state in state.checklist_state]
        return NlvrDecoderState(batch_indices=batch_indices,
                                action_history=action_histories,
                                score=scores,
                                rnn_state=rnn_states,
                                grammar_state=grammar_states,
                                action_embeddings=states[0].action_embeddings,
                                action_indices=states[0].action_indices,
                                possible_actions=states[0].possible_actions,
                                worlds=states[0].worlds,
                                label_strings=states[0].label_strings,
                                checklist_state=checklist_states)
