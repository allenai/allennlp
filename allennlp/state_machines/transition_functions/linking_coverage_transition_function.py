from collections import defaultdict
from typing import Any, Dict, List, Tuple

from overrides import overrides

import torch
from torch.nn import Parameter

from allennlp.common.checks import check_dimensions_match
from allennlp.modules import Attention, FeedForward
from allennlp.nn import Activation
from allennlp.state_machines.states import CoverageState, ChecklistStatelet
from allennlp.state_machines.transition_functions.coverage_transition_function import CoverageTransitionFunction


class LinkingCoverageTransitionFunction(CoverageTransitionFunction):
    """
    Combines both linking and coverage on top of the ``BasicTransitionFunction`` (which is just an
    LSTM decoder with attention).  This adds the ability to consider `linked` actions in addition
    to global (embedded) actions, and it adds a coverage penalty over the `output action sequence`,
    combining the :class:`LinkingTransitionFunction` with the :class:`CoverageTransitionFunction`.

    The one thing that's unique to this class is how the coverage penalty interacts with linked
    actions.  Instead of boosting the action's embedding, as we do in the
    ``CoverageTransitionFunction``, we boost the action's logit directly (as there is no action
    embedding for linked actions).

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
                 mixture_feedforward: FeedForward = None,
                 dropout: float = 0.0) -> None:
        super().__init__(encoder_output_dim=encoder_output_dim,
                         action_embedding_dim=action_embedding_dim,
                         input_attention=input_attention,
                         activation=activation,
                         add_action_bias=add_action_bias,
                         dropout=dropout)
        self._linked_checklist_multiplier = Parameter(torch.FloatTensor([1.0]))
        self._mixture_feedforward = mixture_feedforward

        if mixture_feedforward is not None:
            check_dimensions_match(encoder_output_dim, mixture_feedforward.get_input_dim(),
                                   "hidden state embedding dim", "mixture feedforward input dim")
            check_dimensions_match(mixture_feedforward.get_output_dim(), 1,
                                   "mixture feedforward output dim", "dimension for scalar value")

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
            action_ids: List[int] = []
            if "global" in instance_actions:
                action_embeddings, output_action_embeddings, embedded_actions = instance_actions['global']

                # This embedding addition the only difference between the logic here and the
                # corresponding logic in the super class.
                embedding_addition = self._get_predicted_embedding_addition(state.checklist_state[group_index],
                                                                            embedded_actions,
                                                                            action_embeddings)
                addition = embedding_addition * self._checklist_multiplier
                predicted_action_embedding = predicted_action_embedding + addition

                # This is just a matrix product between a (num_actions, embedding_dim) matrix and an
                # (embedding_dim, 1) matrix.
                embedded_action_logits = action_embeddings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
                action_ids += embedded_actions
            else:
                embedded_action_logits = None
                output_action_embeddings = None

            if 'linked' in instance_actions:
                linking_scores, type_embeddings, linked_actions = instance_actions['linked']
                action_ids += linked_actions
                # (num_question_tokens, 1)
                linked_action_logits = linking_scores.mm(attention_weights[group_index].unsqueeze(-1)).squeeze(-1)

                linked_logits_addition = self._get_linked_logits_addition(state.checklist_state[group_index],
                                                                          linked_actions,
                                                                          linked_action_logits)

                addition = linked_logits_addition * self._linked_checklist_multiplier
                linked_action_logits = linked_action_logits + addition

                # The `output_action_embeddings` tensor gets used later as the input to the next
                # decoder step.  For linked actions, we don't have any action embedding, so we use
                # the entity type instead.
                if output_action_embeddings is None:
                    output_action_embeddings = type_embeddings
                else:
                    output_action_embeddings = torch.cat([output_action_embeddings, type_embeddings], dim=0)

                if self._mixture_feedforward is not None:
                    # The linked and global logits are combined with a mixture weight to prevent the
                    # linked_action_logits from dominating the embedded_action_logits if a softmax
                    # was applied on both together.
                    mixture_weight = self._mixture_feedforward(hidden_state[group_index])
                    mix1 = torch.log(mixture_weight)
                    mix2 = torch.log(1 - mixture_weight)

                    entity_action_probs = torch.nn.functional.log_softmax(linked_action_logits, dim=-1) + mix1
                    if embedded_action_logits is None:
                        current_log_probs = entity_action_probs
                    else:
                        embedded_action_probs = torch.nn.functional.log_softmax(embedded_action_logits,
                                                                                dim=-1) + mix2
                        current_log_probs = torch.cat([embedded_action_probs, entity_action_probs], dim=-1)
                else:
                    if embedded_action_logits is None:
                        current_log_probs = torch.nn.functional.log_softmax(linked_action_logits, dim=-1)
                    else:
                        action_logits = torch.cat([embedded_action_logits, linked_action_logits], dim=-1)
                        current_log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
            else:
                current_log_probs = torch.nn.functional.log_softmax(embedded_action_logits, dim=-1)

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

    @staticmethod
    def _get_linked_logits_addition(checklist_state: ChecklistStatelet,
                                    action_ids: List[int],
                                    action_logits: torch.Tensor) -> torch.Tensor:
        """
        Gets the logits of desired terminal actions yet to be produced by the decoder, and
        returns them for the decoder to add to the prior action logits, biasing the model towards
        predicting missing linked actions.
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

        # Shape: (num_current_actions,).  This is the sum of the action embeddings that we want
        # the model to prefer.
        logit_addition = action_logits * actions_to_encourage
        return logit_addition
