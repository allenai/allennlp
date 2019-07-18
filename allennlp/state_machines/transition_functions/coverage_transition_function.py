from collections import defaultdict
from typing import Any, Dict, List, Tuple

from overrides import overrides

import torch
from torch.nn import Parameter

from allennlp.modules import Attention
from allennlp.nn import Activation
from allennlp.state_machines.states import CoverageState, ChecklistStatelet
from allennlp.state_machines.transition_functions.basic_transition_function import BasicTransitionFunction


class CoverageTransitionFunction(BasicTransitionFunction):
    """
    Adds a coverage penalty to the ``BasicTransitionFunction`` (which is just an LSTM decoder with
    attention).  This coverage penalty is on the `output action sequence`, and requires an
    externally-computed `agenda` of actions that are expected to be produced during decoding, and
    encourages the model to select actions on that agenda.

    The way that we encourage the model to select actions on the agenda is that we add the
    embeddings for actions on the agenda (that are available at this decoding step and haven't yet
    been taken) to the predicted action embedding.  We weight that addition by a learned multiplier
    that gets initialized to 1.

    Parameters
    ----------
    encoder_output_dim : ``int``
    action_embedding_dim : ``int``
    input_attention : ``Attention``
    activation : ``Activation``, optional (default=relu)
        The activation that gets applied to the decoder LSTM input and to the action query.
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
                 activation: Activation = Activation.by_name('relu')(),
                 add_action_bias: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__(encoder_output_dim=encoder_output_dim,
                         action_embedding_dim=action_embedding_dim,
                         input_attention=input_attention,
                         activation=activation,
                         add_action_bias=add_action_bias,
                         dropout=dropout)
        # See the class docstring for a description of what this does.
        self._checklist_multiplier = Parameter(torch.FloatTensor([1.0]))

    @overrides
    def _compute_action_probabilities(self,  # type: ignore
                                      state: CoverageState,
                                      hidden_state: torch.Tensor,
                                      attention_weights: torch.Tensor,
                                      predicted_action_embeddings: torch.Tensor
                                     ) -> Dict[int, List[Tuple[int, Any, Any, Any, List[int]]]]:
        # In this section we take our predicted action embedding and compare it to the available
        # actions in our current state (which might be different for each group element).  For
        # computing action scores, we'll forget about doing batched / grouped computation, as it
        # adds too much complexity and doesn't speed things up, anyway, with the operations we're
        # doing here.  This means we don't need any action masks, as we'll only get the right
        # lengths for what we're computing.

        group_size = len(state.batch_indices)
        actions = state.get_valid_actions()

        batch_results: Dict[int, List[Tuple[int, Any, Any, Any, List[int]]]] = defaultdict(list)
        for group_index in range(group_size):
            instance_actions = actions[group_index]
            predicted_action_embedding = predicted_action_embeddings[group_index]
            action_embeddings, output_action_embeddings, action_ids = instance_actions['global']

            # This embedding addition the only difference between the logic here and the
            # corresponding logic in the super class.
            embedding_addition = self._get_predicted_embedding_addition(state.checklist_state[group_index],
                                                                        action_ids,
                                                                        action_embeddings)
            addition = embedding_addition * self._checklist_multiplier
            predicted_action_embedding = predicted_action_embedding + addition

            # This is just a matrix product between a (num_actions, embedding_dim) matrix and an
            # (embedding_dim, 1) matrix.
            action_logits = action_embeddings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
            current_log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)

            # This is now the total score for each state after taking each action.  We're going to
            # sort by this later, so it's important that this is the total score, not just the
            # score for the current action.
            log_probs = state.score[group_index] + current_log_probs
            batch_results[state.batch_indices[group_index]].append((group_index,
                                                                    log_probs,
                                                                    current_log_probs,
                                                                    output_action_embeddings,
                                                                    action_ids))
        return batch_results

    def _get_predicted_embedding_addition(self,
                                          checklist_state: ChecklistStatelet,
                                          action_ids: List[int],
                                          action_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Gets the embeddings of desired terminal actions yet to be produced by the decoder, and
        returns their sum for the decoder to add it to the predicted embedding to bias the
        prediction towards missing actions.
        """
        # Our basic approach here will be to figure out which actions we want to bias, by doing
        # some fancy indexing work, then multiply the action embeddings by a mask for those
        # actions, and return the sum of the result.

        # Shape: (num_terminal_actions, 1).  This is 1 if we still want to predict something on the
        # checklist, and 0 otherwise.
        checklist_balance = checklist_state.get_balance().clamp(min=0)

        # (num_terminal_actions, 1)
        actions_in_agenda = checklist_state.terminal_actions
        # (1, num_current_actions)
        action_id_tensor = checklist_balance.new(action_ids).long().unsqueeze(0)
        # Shape: (num_terminal_actions, num_current_actions).  Will have a value of 1 if the
        # terminal action i is our current action j, and a value of 0 otherwise.  Because both sets
        # of actions are free of duplicates, there will be at most one non-zero value per current
        # action, and per terminal action.
        current_agenda_actions = (actions_in_agenda == action_id_tensor).float()

        # Shape: (num_current_actions,).  With the inner multiplication, we remove any current
        # agenda actions that are not in our checklist balance, then we sum over the terminal
        # action dimension, which will have a sum of at most one.  So this will be a 0/1 tensor,
        # where a 1 means to encourage the current action in that position.
        actions_to_encourage = torch.sum(current_agenda_actions * checklist_balance, dim=0)

        # Shape: (action_embedding_dim,).  This is the sum of the action embeddings that we want
        # the model to prefer.
        embedding_addition = torch.sum(action_embeddings * actions_to_encourage.unsqueeze(1),
                                       dim=0,
                                       keepdim=False)

        if self._add_action_bias:
            # If we're adding an action bias, the last dimension of the action embedding is a bias
            # weight.  We don't want this addition to affect the bias (TODO(mattg): or do we?), so
            # we zero out that dimension here.
            embedding_addition[-1] = 0

        return embedding_addition
