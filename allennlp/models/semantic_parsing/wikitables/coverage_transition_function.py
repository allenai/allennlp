from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from overrides import overrides

import torch
from torch.nn import Parameter
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear

from allennlp.common import util as common_util
from allennlp.common.checks import check_dimensions_match
from allennlp.models.semantic_parsing.wikitables.basic_transition_function import BasicTransitionFunction
from allennlp.models.semantic_parsing.wikitables.grammar_based_decoder_state import GrammarBasedDecoderState
from allennlp.modules import Attention, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util, Activation
from allennlp.nn.decoding import ChecklistState, DecoderStep, RnnState


class CoverageTransitionFunction(BasicTransitionFunction):
    """
    Adds a coverage penalty to the ``BasicTransitionFunction`` (which is just an LSTM decoder with
    attention).  This coverage penalty is on the `output action sequence`, and requires an
    externally-computed `agenda` of actions that are expected to be produced during decoding, and
    encourages the model to select actions on that agenda.

    Parameters
    ----------
    encoder_output_dim : ``int``
    action_embedding_dim : ``int``
    input_attention : ``Attention``
    num_start_types : ``int``
    activation : ``Activation``, optional (default=relu)
        The activation that gets applied to the decoder LSTM input and to the action query.
    predict_start_type_separately : ``bool``, optional (default=True)
        If ``True``, we will predict the initial action (which is typically the base type of the
        logical form) using a different mechanism than our typical action decoder.  We basically
        just do a projection of the hidden state, and don't update the decoder RNN.
    add_action_bias : ``bool``, optional (default=True)
        If ``True``, there has been a bias dimension added to the embedding of each action, which
        gets used when predicting the next action.  We add a dimension of ones to our predicted
        action vector in this case to account for that.
    dropout : ``float`` (optional, default=0.0)
    """
    def __init__(self,
                 encoder_output_dim: int,
                 action_embedding_dim: int,
                 input_attention: Attention,
                 num_start_types: int,
                 activation: Activation = Activation.by_name('relu')(),
                 predict_start_type_separately: bool = True,
                 add_action_bias: bool = True,
                 mixture_feedforward: FeedForward = None,
                 dropout: float = 0.0,
                 unlinked_terminal_indices: List[int] = None) -> None:
        super().__init__(encoder_output_dim=encoder_output_dim,
                         action_embedding_dim=action_embedding_dim,
                         input_attention=input_attention,
                         num_start_types=num_start_types,
                         activation=activation,
                         predict_start_type_separately=predict_start_type_separately,
                         add_action_bias=add_action_bias,
                         mixture_feedforward=mixture_feedforward,
                         dropout=dropout)
        self._checklist_multiplier = Parameter(torch.FloatTensor([1.0]))

        self._unlinked_terminal_indices = unlinked_terminal_indices

    def _compute_action_probabilities(self,
                                      state: GrammarBasedDecoderState,
                                      hidden_state: torch.Tensor,
                                      attention_weights: torch.Tensor,
                                      predicted_action_embeddings: torch.Tensor
                                      ) -> Dict[int, List[Tuple[int, Any, Any, List[int]]]]:
        # In this section we take our predicted action embedding and compare it to the available
        # actions in our current state (which might be different for each group element).  For
        # computing action scores, we'll forget about doing batched / grouped computation, as it
        # adds too much complexity and doesn't speed things up, anyway, with the operations we're
        # doing here.  This means we don't need any action masks, as we'll only get the right
        # lengths for what we're computing.

        group_size = len(state.batch_indices)
        actions = state.get_valid_actions()

        batch_results: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor, List[int]]]] = defaultdict(list)
        for group_index in range(group_size):
            instance_actions = actions[group_index]
            predicted_action_embedding = predicted_action_embeddings[group_index]
            action_embeddings, output_action_embeddings, action_ids = instance_actions['global']
            checklist_balance = self._get_checklist_balance(state.checklist_state[group_index])
            embedding_addition = self._get_predicted_embedding_addition(state,
                                                                        group_index,
                                                                        self._unlinked_terminal_indices,
                                                                        unlinked_balance)
            addition = embedding_addition * self._unlinked_checklist_multiplier
            predicted_action_embedding = predicted_action_embedding + addition

            # This is just a matrix product between a (num_actions, embedding_dim) matrix and an
            # (embedding_dim, 1) matrix.
            action_logits = action_embeddings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
            # This is now the total score for each state after taking each action.  We're going to
            # sort by this later, so it's important that this is the total score, not just the
            # score for the current action.
            log_probs = state.score[group_index] + current_log_probs
            batch_results[state.batch_indices[group_index]].append((group_index,
                                                                    log_probs,
                                                                    output_action_embeddings,
                                                                    action_ids))
        return batch_results

    @staticmethod
    def _get_checklist_balance(checklist_state: ChecklistState,
                               unlinked_terminal_indices: List[int],
                               actions_to_link: List[List[int]]) -> Tuple[torch.FloatTensor,
                                                                          torch.FloatTensor]:
        # TODO(mattg): This can be dramatically simplified and merged with
        # get_predicted_embedding_addtion.  All we want to do is take the actions that are still on
        # the checklist and add their embeddings.  Probably easier to do this on CPU, too.  For
        # linked actions, too, when we get to that, we can make this _way_ simpler - again, we just
        # want to get the linked actions and bump up their logits.

        # This holds a checklist balance for this state. The balance is a float vector containing
        # just 1s and 0s showing which of the items are filled. We clamp the min at 0 to ignore the
        # number of times an action is taken. The value at an index will be 1 iff the target wants
        # an unmasked action to be taken, and it is not yet taken.
        checklist_balance = torch.clamp(state.checklist_state[group_index].get_balance(), min=0.0)

        # We now need to split the ``checklist_balance`` into two tensors, one corresponding to
        # linked actions and the other to unlinked actions because they affect the output action
        # logits differently. We use ``unlinked_terminal_indices`` and ``actions_to_link`` to do that, but
        # the indices in those lists are indices of all actions, and the checklist balance
        # corresponds only to the terminal actions.
        # To make things more confusing, ``actions_to_link`` has batch action indices, and
        # ``unlinked_terminal_indices`` has global action indices.
        mapped_actions_to_link = []
        mapped_actions_to_embed = []
        # Mapping from batch action indices to checklist indices for each instance in group.
        batch_actions_to_checklist = [checklist_state.terminal_indices_dict
                                      for checklist_state in state.checklist_state]
        for group_index, batch_index in enumerate(state.batch_indices):
            instance_mapped_embedded_actions = []
            for action in unlinked_terminal_indices:
                batch_action_index = state.global_to_batch_action_indices[(batch_index, action)]
                if batch_action_index in batch_actions_to_checklist[group_index]:
                    checklist_index = batch_actions_to_checklist[group_index][batch_action_index]
                else:
                    # This means that the embedded action is not a terminal, because the checklist
                    # indices only correspond to terminal actions.
                    checklist_index = -1
                instance_mapped_embedded_actions.append(checklist_index)
            mapped_actions_to_embed.append(instance_mapped_embedded_actions)
        # We don't need to pad the unlinked actions because they're all currently the
        # same size as ``unlinked_terminal_indices``.
        unlinked_action_indices = checklist_balance.new_tensor(mapped_actions_to_embed, dtype=torch.long)
        unlinked_actions_mask = (unlinked_action_indices != -1).long()
        # torch.gather would complain if the indices are -1. So making them all 0 now. We'll use the
        # mask again on the balances.
        unlinked_action_indices = unlinked_action_indices * unlinked_actions_mask

        unlinked_checklist_balance = torch.gather(checklist_balance, 1, unlinked_action_indices)
        unlinked_checklist_balance = unlinked_checklist_balance * unlinked_actions_mask.float()
        # If actions_to_link is None, it means that all the valid actions in the current state need
        # to be embedded. We simply return None for checklist balance corresponding to linked
        # actions then.
        linked_checklist_balance = None
        if actions_to_link:
            for group_index, instance_actions_to_link in enumerate(actions_to_link):
                mapped_actions_to_link.append([batch_actions_to_checklist[group_index][action]
                                               for action in instance_actions_to_link])
            # We need to pad the linked action indices before we use them to gather appropriate balances.
            # Some of the indices may be 0s. So we need to make the padding index -1.
            max_num_linked_actions = max([len(indices) for indices in mapped_actions_to_link])
            padded_actions_to_link = [common_util.pad_sequence_to_length(indices,
                                                                         max_num_linked_actions,
                                                                         default_value=lambda: -1)
                                      for indices in mapped_actions_to_link]
            linked_action_indices = checklist_balance.new_tensor(padded_actions_to_link, dtype=torch.long)
            linked_actions_mask = (linked_action_indices != -1).long()
            linked_action_indices = linked_action_indices * linked_actions_mask
            linked_checklist_balance = torch.gather(checklist_balance, 1, linked_action_indices)
            linked_checklist_balance = linked_checklist_balance * linked_actions_mask.float()
        return linked_checklist_balance, unlinked_checklist_balance

    @staticmethod
    def _get_predicted_embedding_addition(state: GrammarBasedDecoderState,
                                          unlinked_terminal_indices: List[int],
                                          unlinked_checklist_balance: torch.Tensor) -> torch.Tensor:
        """
        Gets the embeddings of desired unlinked terminal actions yet to be produced by the decoder,
        and returns their sum for the decoder to add it to the predicted embedding to bias the
        prediction towards missing actions.
        """
        # TODO(mattg): handle the action bias correctly here
        # (group_size, num_unlinked_actions, 1)
        unlinked_balance = unlinked_checklist_balance.unsqueeze(2)
        group_size = len(state.batch_indices)
        action_embedding_dim = state.action_embeddings.size(-1)
        num_terminals = len(unlinked_terminal_indices)
        group_terminal_indices = [unlinked_terminal_indices for _ in range(group_size)]
        # (group_size, num_unlinked_actions)
        terminal_indices_tensor = state.score[0].new_tensor(group_terminal_indices, dtype=torch.long)
        flattened_terminal_indices = terminal_indices_tensor.view(-1)
        flattened_action_embeddings = state.action_embeddings.index_select(0,
                                                                           flattened_terminal_indices)
        terminal_embeddings = flattened_action_embeddings.view(group_size, num_terminals, action_embedding_dim)
        checklist_balance_embeddings = terminal_embeddings * unlinked_balance
        # (group_size, action_embedding_dim)
        return checklist_balance_embeddings.sum(1)
