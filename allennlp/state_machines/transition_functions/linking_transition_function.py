from collections import defaultdict
from typing import Any, Dict, List, Tuple

from overrides import overrides

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.modules import Attention, FeedForward
from allennlp.nn import Activation
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.transition_functions.basic_transition_function import BasicTransitionFunction


class LinkingTransitionFunction(BasicTransitionFunction):
    """
    This transition function adds the ability to consider `linked` actions to the
    ``BasicTransitionFunction`` (which is just an LSTM decoder with attention).  These actions are
    potentially unseen at training time, so we need to handle them without requiring the action to
    have an embedding.  Instead, we rely on a `linking score` between each action and the words in
    the question/utterance, and use these scores, along with the attention, to do something similar
    to a copy mechanism when producing these actions.

    When both linked and global (embedded) actions are available, we need some way to compare the
    scores for these two sets of actions.  The original WikiTableQuestion semantic parser just
    concatenated the logits together before doing a joint softmax, but this is quite brittle,
    because the logits might have quite different scales.  So we have the option here of predicting
    a mixture probability between two independently normalized distributions.

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
    mixture_feedforward : ``FeedForward`` optional (default=None)
        If given, we'll use this to compute a mixture probability between global actions and linked
        actions given the hidden state at every timestep of decoding, instead of concatenating the
        logits for both (where the logits may not be compatible with each other).
    dropout : ``float`` (optional, default=0.0)
    num_layers: ``int`` (optional, default=1)
        The number of layers in the decoder LSTM.
    """
    def __init__(self,
                 encoder_output_dim: int,
                 action_embedding_dim: int,
                 input_attention: Attention,
                 activation: Activation = Activation.by_name('relu')(),
                 add_action_bias: bool = True,
                 mixture_feedforward: FeedForward = None,
                 dropout: float = 0.0,
                 num_layers: int = 1) -> None:
        super().__init__(encoder_output_dim=encoder_output_dim,
                         action_embedding_dim=action_embedding_dim,
                         input_attention=input_attention,
                         activation=activation,
                         add_action_bias=add_action_bias,
                         dropout=dropout,
                         num_layers=num_layers)
        self._mixture_feedforward = mixture_feedforward

        if mixture_feedforward is not None:
            check_dimensions_match(encoder_output_dim, mixture_feedforward.get_input_dim(),
                                   "hidden state embedding dim", "mixture feedforward input dim")
            check_dimensions_match(mixture_feedforward.get_output_dim(), 1,
                                   "mixture feedforward output dim", "dimension for scalar value")

    @overrides
    def _compute_action_probabilities(self,
                                      state: GrammarBasedState,
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
            embedded_actions: List[int] = []

            output_action_embeddings = None
            embedded_action_logits = None
            current_log_probs = None

            if 'global' in instance_actions:
                action_embeddings, output_action_embeddings, embedded_actions = instance_actions['global']
                # This is just a matrix product between a (num_actions, embedding_dim) matrix and an
                # (embedding_dim, 1) matrix.
                embedded_action_logits = action_embeddings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
                action_ids = embedded_actions

            if 'linked' in instance_actions:
                linking_scores, type_embeddings, linked_actions = instance_actions['linked']
                action_ids = embedded_actions + linked_actions
                # (num_question_tokens, 1)
                linked_action_logits = linking_scores.mm(attention_weights[group_index].unsqueeze(-1)).squeeze(-1)

                # The `output_action_embeddings` tensor gets used later as the input to the next
                # decoder step.  For linked actions, we don't have any action embedding, so we use
                # the entity type instead.
                if output_action_embeddings is not None:
                    output_action_embeddings = torch.cat([output_action_embeddings, type_embeddings], dim=0)
                else:
                    output_action_embeddings = type_embeddings

                if self._mixture_feedforward is not None:
                    # The linked and global logits are combined with a mixture weight to prevent the
                    # linked_action_logits from dominating the embedded_action_logits if a softmax
                    # was applied on both together.
                    mixture_weight = self._mixture_feedforward(hidden_state[group_index])
                    mix1 = torch.log(mixture_weight)
                    mix2 = torch.log(1 - mixture_weight)

                    entity_action_probs = torch.nn.functional.log_softmax(linked_action_logits, dim=-1) + mix1
                    if embedded_action_logits is not None:
                        embedded_action_probs = torch.nn.functional.log_softmax(embedded_action_logits,
                                                                                dim=-1) + mix2
                        current_log_probs = torch.cat([embedded_action_probs, entity_action_probs], dim=-1)
                    else:
                        current_log_probs = entity_action_probs
                else:
                    if embedded_action_logits is not None:
                        action_logits = torch.cat([embedded_action_logits, linked_action_logits], dim=-1)
                    else:
                        action_logits = linked_action_logits
                    current_log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
            else:
                action_logits = embedded_action_logits
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
