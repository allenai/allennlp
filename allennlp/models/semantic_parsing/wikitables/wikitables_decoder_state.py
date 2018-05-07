from typing import Dict, List, Tuple

import torch

from allennlp.semparse.worlds import WikiTablesWorld
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
    output_action_embeddings : ``torch.Tensor``
        The global output action embeddings tensor.  Has shape ``(num_global_embeddable_actions,
        action_embedding_dim)``.
    action_biases : ``torch.Tensor``
        A vector of biases for each action.  Has shape ``(num_global_embeddable_actions, 1)``.
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
    world : ``List[WikiTablesWorld]``, optional (default=None)
        The worlds corresponding to elements in the batch. We store them here because they're required
        for executing logical forms to determine costs while training, if we're learning to search.
        Otherwise, they're not required. Note that the worlds are batched, and they will be passed
        around unchanged during the decoding process.
    example_lisp_string : ``List[str]``, optional (default=None)
        The lisp strings that come from example files. They're also required for evaluating logical
        forms only if we're learning to search. These too are batched, and will be passed around
        unchanged.
    terminal_actions : ``List[torch.Tensor]``, optional (default=None)
        Each element in the list is a vector containing the indices of terminal actions. They are
        needed for computing checklists for next states, only if this state is made while training a
        parser without logical forms.
    checklist_target : ``List[torch.LongTensor]``, optional (default=None)
        List of targets corresponding to agendas that indicate the states we want the checklists to
        ideally be. Each element in this list is the same size as the corresponding element in
        ``terminal_actions``, and it contains 1 for each corresponding action in the relevant
        actions list that we want to see in the final logical form, and 0 for each corresponding
        action that we do not. Needed only if this state is being used while training a parser
        without logical forms.
    checklist_masks : ``List[torch.Tensor]``, optional (default=None)
        Masks corresponding to ``terminal_actions``, indicating which of those actions are relevant
        for checklist computation. For example, if the parser is penalizing non-agenda terminal
        actions, all the terminal actions are relevant. Needed only if this state is being used
        while training a parser without logical forms.
    checklist : ``List[Variable]``, optional (default=None)
        A checklist for each instance indicating how many times each action in its agenda has
        been chosen previously. It contains the actual counts of the agenda actions. Needed only if
        this state is being used while training a parser without logical forms.
    """
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor],
                 rnn_state: List[RnnState],
                 grammar_state: List[GrammarState],
                 action_embeddings: torch.Tensor,
                 output_action_embeddings: torch.Tensor,
                 action_biases: torch.Tensor,
                 action_indices: Dict[Tuple[int, int], int],
                 possible_actions: List[List[ProductionRuleArray]],
                 flattened_linking_scores: torch.FloatTensor,
                 actions_to_entities: Dict[Tuple[int, int], int],
                 entity_types: Dict[int, int],
                 world: List[WikiTablesWorld] = None,
                 example_lisp_string: List[str] = None,
                 terminal_actions: List[torch.Tensor] = None,
                 checklist_target: List[torch.Tensor] = None,
                 checklist_masks: List[torch.Tensor] = None,
                 checklist: List[torch.Tensor] = None,
                 debug_info: List = None) -> None:
        super(WikiTablesDecoderState, self).__init__(batch_indices, action_history, score)
        self.rnn_state = rnn_state
        self.grammar_state = grammar_state
        self.action_embeddings = action_embeddings
        self.output_action_embeddings = output_action_embeddings
        self.action_biases = action_biases
        self.action_indices = action_indices
        self.possible_actions = possible_actions
        self.flattened_linking_scores = flattened_linking_scores
        self.actions_to_entities = actions_to_entities
        self.entity_types = entity_types
        self.world = world
        self.example_lisp_string = example_lisp_string
        self.terminal_actions = terminal_actions
        self.checklist_target = checklist_target
        self.checklist_mask = checklist_masks
        self.checklist = checklist
        self.debug_info = debug_info

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
        if states[0].terminal_actions is not None:
            terminal_actions = [actions for state in states for actions in state.terminal_actions]
            checklist_target = [target_list for state in states for target_list in
                                state.checklist_target]
            checklist_masks = [mask for state in states for mask in state.checklist_mask]
            checklist = [checklist_list for state in states for checklist_list in state.checklist]
        else:
            terminal_actions = None
            checklist_target = None
            checklist_masks = None
            checklist = None
        return WikiTablesDecoderState(batch_indices=batch_indices,
                                      action_history=action_histories,
                                      score=scores,
                                      rnn_state=rnn_states,
                                      grammar_state=grammar_states,
                                      action_embeddings=states[0].action_embeddings,
                                      output_action_embeddings=states[0].output_action_embeddings,
                                      action_biases=states[0].action_biases,
                                      action_indices=states[0].action_indices,
                                      possible_actions=states[0].possible_actions,
                                      flattened_linking_scores=states[0].flattened_linking_scores,
                                      actions_to_entities=states[0].actions_to_entities,
                                      entity_types=states[0].entity_types,
                                      world=states[0].world,
                                      example_lisp_string=states[0].example_lisp_string,
                                      terminal_actions=terminal_actions,
                                      checklist_target=checklist_target,
                                      checklist_masks=checklist_masks,
                                      checklist=checklist,
                                      debug_info=debug_info)
