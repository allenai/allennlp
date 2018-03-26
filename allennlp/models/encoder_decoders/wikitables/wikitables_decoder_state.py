from typing import Dict, List, Tuple

import torch

from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.nn.decoding import DecoderState, GrammarState, RnnState


# This syntax is pretty weird and ugly, but it's necessary to make mypy happy with the API that
# we've defined.  We're using generics to make the type of `combine_states` come out right.  See
# the note in `nn.decoding.decoder_state.py` for a little more detail.
class WikiTablesDecoderState(DecoderState['WikiTablesDecoderState']):
    """
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
    flattened_linking_scores : ``torch.FloatTensor``
        Linking scores between table entities and question tokens.  The unflattened version has
        shape ``(batch_size, num_entities, num_question_tokens)``, though this version is flattened
        to have shape ``(batch_size * num_entities, num_question_tokens)``, for easier lookups with
        ``index_select``.
    actions_to_entities : ``Dict[Tuple[int, int], int]``
        A mapping from ``(batch_index, action_index)`` to ``batch_size * num_entities``, for
        actions that are terminal entity productions.
    entity_types : ``Dict[int, int]``
        A mapping from flattened entity indices (same as the `values` in the
        ``actions_to_entities`` dictionary) to entity type indices.  This represents what type each
        entity has, which we will use for getting type embeddings in certain circumstances.
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
                 flattened_linking_scores: torch.FloatTensor,
                 actions_to_entities: Dict[Tuple[int, int], int],
                 entity_types: Dict[int, int],
                 debug_info: List = None) -> None:
        super(WikiTablesDecoderState, self).__init__(batch_indices, action_history, score)
        self.rnn_state = rnn_state
        self.grammar_state = grammar_state
        self.action_embeddings = action_embeddings
        self.action_indices = action_indices
        self.possible_actions = possible_actions
        self.flattened_linking_scores = flattened_linking_scores
        self.actions_to_entities = actions_to_entities
        self.entity_types = entity_types
        self.debug_info = debug_info

    def get_valid_actions(self) -> List[List[int]]:
        """
        Returns a list of valid actions for each element of the group.
        """
        return [state.get_valid_actions() for state in self.grammar_state]

    def is_finished(self) -> bool:
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_state[0].is_finished()

    @classmethod
    def combine_states(cls, states: List['WikiTablesDecoderState']) -> 'WikiTablesDecoderState':
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        rnn_states = [rnn_state for state in states for rnn_state in state.rnn_state]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        if states[0].debug_info is not None:
            debug_info = [debug_info for state in states for debug_info in state.debug_info]
        else:
            debug_info = None
        return WikiTablesDecoderState(batch_indices=batch_indices,
                                      action_history=action_histories,
                                      score=scores,
                                      rnn_state=rnn_states,
                                      grammar_state=grammar_states,
                                      action_embeddings=states[0].action_embeddings,
                                      action_indices=states[0].action_indices,
                                      possible_actions=states[0].possible_actions,
                                      flattened_linking_scores=states[0].flattened_linking_scores,
                                      actions_to_entities=states[0].actions_to_entities,
                                      entity_types=states[0].entity_types,
                                      debug_info=debug_info)
