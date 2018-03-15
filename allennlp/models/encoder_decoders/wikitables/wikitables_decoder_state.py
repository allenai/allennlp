from typing import Dict, List, Tuple

import torch

from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.nn.decoding import DecoderState
from allennlp.semparse.type_declarations import GrammarState


# This syntax is pretty weird and ugly, but it's necessary to make mypy happy with the API that
# we've defined.  We're using generics to make the type of `combine_states` come out right.  See
# the note in `nn.decoding.decoder_state.py` for a little more detail.
class WikiTablesDecoderState(DecoderState['WikiTablesDecoderState']):
    """
    TODO(mattg): This class is a mess!  We need to figure out a better way to pass around and
    update this state.  There are too many things going on here, and it's getting out of control.

    Parameters
    ----------
    batch_indices : ``List[int]``
        Passed to super class; see docs there.
    action_history : ``List[List[int]]``
        Passed to super class; see docs there.
    score : ``List[torch.Tensor]``
        Passed to super class; see docs there.
    hidden_state : ``List[torch.Tensor]``
        This holds the LSTM hidden state for each element of the group.  Each tensor has shape
        ``(decoder_output_dim,)``.
    memory_cell : ``List[torch.Tensor]``
        This holds the LSTM memory cell for each element of the group.  Each tensor has shape
        ``(decoder_output_dim,)``.
    previous_action_embedding : ``List[torch.Tensor]``
        This holds the embedding for the action we took at the last timestep (which gets input to
        the decoder).  Each tensor has shape ``(action_embedding_dim,)``.
    attended_question : ``List[torch.Tensor]``
        This holds the attention-weighted sum over the question representations that we computed in
        the previous timestep, for each element in the group.  We keep this as part of the state
        because we use the previous attention as part of our decoder cell update.  Each tensor in
        this list has shape ``(encoder_output_dim,)``.
    grammar_state : ``List[GrammarState]``
        This hold the current grammar state for each element of the group.  The ``GrammarState``
        keeps track of which actions are currently valid.
    encoder_outputs : ``List[torch.Tensor]``
        A list of variables, each of shape ``(question_length, encoder_output_dim)``, containing
        the encoder outputs at each timestep.  The list is over batch elements, and we do the input
        this way so we can easily do a ``torch.cat`` on a list of indices into this batched list.

        Note that all of the above lists are of length ``group_size``, while the encoder outputs
        and mask are lists of length ``batch_size``.  We always pass around the encoder outputs and
        mask unmodified, regardless of what's in the grouping for this state.  We'll use the
        ``batch_indices`` for the group to pull pieces out of these lists when we're ready to
        actually do some computation.
    encoder_output_mask : ``List[torch.Tensor]``
        A list of variables, each of shape ``(question_length,)``, containing a mask over question
        tokens for each batch instance.  This is a list over batch elements, for the same reasons
        as above.
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
                 hidden_state: List[torch.Tensor],
                 memory_cell: List[torch.Tensor],
                 previous_action_embedding: List[torch.Tensor],
                 attended_question: List[torch.Tensor],
                 grammar_state: List[GrammarState],
                 encoder_outputs: torch.Tensor,
                 encoder_output_mask: torch.Tensor,
                 action_embeddings: torch.Tensor,
                 action_indices: Dict[Tuple[int, int], int],
                 possible_actions: List[List[ProductionRuleArray]],
                 flattened_linking_scores: torch.FloatTensor,
                 actions_to_entities: Dict[Tuple[int, int], int],
                 entity_types: Dict[int, int],
                 debug_info: List = None) -> None:
        super(WikiTablesDecoderState, self).__init__(batch_indices, action_history, score)
        self.hidden_state = hidden_state
        self.memory_cell = memory_cell
        self.previous_action_embedding = previous_action_embedding
        self.attended_question = attended_question
        self.grammar_state = grammar_state
        self.encoder_outputs = encoder_outputs
        self.encoder_output_mask = encoder_output_mask
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

    # @overrides  - overrides can't handle the generics we're using here, apparently
    def is_finished(self) -> bool:
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_state[0].is_finished()

    @classmethod
    # @overrides  - overrides can't handle the generics we're using here, apparently
    def combine_states(cls, states: List['WikiTablesDecoderState']) -> 'WikiTablesDecoderState':
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        hidden_states = [hidden_state for state in states for hidden_state in state.hidden_state]
        memory_cells = [memory_cell for state in states for memory_cell in state.memory_cell]
        previous_action = [action for state in states for action in state.previous_action_embedding]
        attended_question = [attended for state in states for attended in state.attended_question]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        if states[0].debug_info is not None:
            debug_info = [debug_info for state in states for debug_info in state.debug_info]
        else:
            debug_info = None
        return WikiTablesDecoderState(batch_indices=batch_indices,
                                      action_history=action_histories,
                                      score=scores,
                                      hidden_state=hidden_states,
                                      memory_cell=memory_cells,
                                      previous_action_embedding=previous_action,
                                      attended_question=attended_question,
                                      grammar_state=grammar_states,
                                      encoder_outputs=states[0].encoder_outputs,
                                      encoder_output_mask=states[0].encoder_output_mask,
                                      action_embeddings=states[0].action_embeddings,
                                      action_indices=states[0].action_indices,
                                      possible_actions=states[0].possible_actions,
                                      flattened_linking_scores=states[0].flattened_linking_scores,
                                      actions_to_entities=states[0].actions_to_entities,
                                      entity_types=states[0].entity_types,
                                      debug_info=debug_info)
