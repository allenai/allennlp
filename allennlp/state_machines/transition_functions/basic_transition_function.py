from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from overrides import overrides

import torch
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear

from allennlp.modules import Attention
from allennlp.nn import util, Activation
from allennlp.state_machines.states import RnnStatelet, GrammarBasedState
from allennlp.state_machines.transition_functions.transition_function import TransitionFunction


class BasicTransitionFunction(TransitionFunction[GrammarBasedState]):
    """
    This is a typical transition function for a state-based decoder.  We use an LSTM to track
    decoder state, and at every timestep we compute an attention over the input question/utterance
    to help in selecting the action.  All actions have an embedding, and we use a dot product
    between a predicted action embedding and the allowed actions to compute a distribution over
    actions at each timestep.

    We allow the first action to be predicted separately from everything else.  This is optional,
    and is because that's how the original WikiTableQuestions semantic parser was written.  The
    intuition is that maybe you want to predict the type of your output program outside of the
    typical LSTM decoder (or maybe Jayant just didn't realize this could be treated as another
    action...).

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
                 dropout: float = 0.0) -> None:
        super().__init__()
        self._input_attention = input_attention
        self._add_action_bias = add_action_bias
        self._activation = activation

        self._num_start_types = num_start_types
        self._predict_start_type_separately = predict_start_type_separately
        if predict_start_type_separately:
            self._start_type_predictor = Linear(encoder_output_dim, num_start_types)
        else:
            self._start_type_predictor = None

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        output_dim = encoder_output_dim
        input_dim = output_dim
        # Our decoder input will be the concatenation of the decoder hidden state and the previous
        # action embedding, and we'll project that down to the decoder's `input_dim`, which we
        # arbitrarily set to be the same as `output_dim`.
        self._input_projection_layer = Linear(output_dim + action_embedding_dim, input_dim)
        # Before making a prediction, we'll compute an attention over the input given our updated
        # hidden state. Then we concatenate those with the decoder state and project to
        # `action_embedding_dim` to make a prediction.
        self._output_projection_layer = Linear(output_dim + encoder_output_dim, action_embedding_dim)

        self._decoder_cell = LSTMCell(input_dim, output_dim)

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    @overrides
    def take_step(self,
                  state: GrammarBasedState,
                  max_actions: int = None,
                  allowed_actions: List[Set[int]] = None) -> List[GrammarBasedState]:
        if self._predict_start_type_separately and not state.action_history[0]:
            # The wikitables parser did something different when predicting the start type, which
            # is our first action.  So in this case we break out into a different function.  We'll
            # ignore max_actions on our first step, assuming there aren't that many start types.
            return self._take_first_step(state, allowed_actions)

        # Taking a step in the decoder consists of three main parts.  First, we'll construct the
        # input to the decoder and update the decoder's hidden state.  Second, we'll use this new
        # hidden state (and maybe other information) to predict an action.  Finally, we will
        # construct new states for the next step.  Each new state corresponds to one valid action
        # that can be taken from the current state, and they are ordered by their probability of
        # being selected.

        updated_state = self._update_decoder_state(state)
        batch_results = self._compute_action_probabilities(state,
                                                           updated_state['hidden_state'],
                                                           updated_state['attention_weights'],
                                                           updated_state['predicted_action_embeddings'])
        new_states = self._construct_next_states(state,
                                                 updated_state,
                                                 batch_results,
                                                 max_actions,
                                                 allowed_actions)

        return new_states

    def _update_decoder_state(self, state: GrammarBasedState) -> Dict[str, torch.Tensor]:
        # For updating the decoder, we're doing a bunch of tensor operations that can be batched
        # without much difficulty.  So, we take all group elements and batch their tensors together
        # before doing these decoder operations.

        group_size = len(state.batch_indices)
        attended_question = torch.stack([rnn_state.attended_input for rnn_state in state.rnn_state])
        hidden_state = torch.stack([rnn_state.hidden_state for rnn_state in state.rnn_state])
        memory_cell = torch.stack([rnn_state.memory_cell for rnn_state in state.rnn_state])
        previous_action_embedding = torch.stack([rnn_state.previous_action_embedding
                                                 for rnn_state in state.rnn_state])

        # (group_size, decoder_input_dim)
        projected_input = self._input_projection_layer(torch.cat([attended_question,
                                                                  previous_action_embedding], -1))
        decoder_input = self._activation(projected_input)

        hidden_state, memory_cell = self._decoder_cell(decoder_input, (hidden_state, memory_cell))
        hidden_state = self._dropout(hidden_state)

        # (group_size, encoder_output_dim)
        encoder_outputs = torch.stack([state.rnn_state[0].encoder_outputs[i] for i in state.batch_indices])
        encoder_output_mask = torch.stack([state.rnn_state[0].encoder_output_mask[i] for i in state.batch_indices])
        attended_question, attention_weights = self.attend_on_question(hidden_state,
                                                                       encoder_outputs,
                                                                       encoder_output_mask)
        action_query = torch.cat([hidden_state, attended_question], dim=-1)

        # (group_size, action_embedding_dim)
        projected_query = self._activation(self._output_projection_layer(action_query))
        predicted_action_embeddings = self._dropout(projected_query)
        if self._add_action_bias:
            # NOTE: It's important that this happens right before the dot product with the action
            # embeddings.  Otherwise this isn't a proper bias.  We do it here instead of right next
            # to the `.mm` below just so we only do it once for the whole group.
            ones = predicted_action_embeddings.new([[1] for _ in range(group_size)])
            predicted_action_embeddings = torch.cat([predicted_action_embeddings, ones], dim=-1)
        return {
                'hidden_state': hidden_state,
                'memory_cell': memory_cell,
                'attended_question': attended_question,
                'attention_weights': attention_weights,
                'predicted_action_embeddings': predicted_action_embeddings,
                }

    def _compute_action_probabilities(self,
                                      state: GrammarBasedState,
                                      hidden_state: torch.Tensor,
                                      attention_weights: torch.Tensor,
                                      predicted_action_embeddings: torch.Tensor
                                     ) -> Dict[int, List[Tuple[int, Any, Any, List[int]]]]:
        # We take a couple of extra arguments here because subclasses might use them.
        # pylint: disable=unused-argument,no-self-use

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
                                                                    output_action_embeddings,
                                                                    action_ids))
        return batch_results

    def _construct_next_states(self,
                               state: GrammarBasedState,
                               updated_rnn_state: Dict[str, torch.Tensor],
                               batch_action_probs: Dict[int, List[Tuple[int, Any, Any, List[int]]]],
                               max_actions: int,
                               allowed_actions: List[Set[int]]):
        # pylint: disable=no-self-use

        # We'll yield a bunch of states here that all have a `group_size` of 1, so that the
        # learning algorithm can decide how many of these it wants to keep, and it can just regroup
        # them later, as that's a really easy operation.
        #
        # We first define a `make_state` method, as in the logic that follows we want to create
        # states in a couple of different branches, and we don't want to duplicate the
        # state-creation logic.  This method creates a closure using variables from the method, so
        # it doesn't make sense to pull it out of here.

        # Each group index here might get accessed multiple times, and doing the slicing operation
        # each time is more expensive than doing it once upfront.  These three lines give about a
        # 10% speedup in training time.
        group_size = len(state.batch_indices)
        hidden_state = [x.squeeze(0) for x in updated_rnn_state['hidden_state'].chunk(group_size, 0)]
        memory_cell = [x.squeeze(0) for x in updated_rnn_state['memory_cell'].chunk(group_size, 0)]
        attended_question = [x.squeeze(0) for x in updated_rnn_state['attended_question'].chunk(group_size, 0)]

        def make_state(group_index: int,
                       action: int,
                       new_score: torch.Tensor,
                       action_embedding: torch.Tensor) -> GrammarBasedState:
            new_rnn_state = RnnStatelet(hidden_state[group_index],
                                        memory_cell[group_index],
                                        action_embedding,
                                        attended_question[group_index],
                                        state.rnn_state[group_index].encoder_outputs,
                                        state.rnn_state[group_index].encoder_output_mask)
            batch_index = state.batch_indices[group_index]
            for i, log_probs, _, actions in batch_action_probs[batch_index]:
                if i == group_index:
                    considered_actions = actions
                    probabilities = log_probs.exp().cpu()
                    break
            return state.new_state_from_group_index(group_index,
                                                    action,
                                                    new_score,
                                                    new_rnn_state,
                                                    considered_actions,
                                                    probabilities,
                                                    updated_rnn_state['attention_weights'])

        new_states = []
        for _, results in batch_action_probs.items():
            if allowed_actions and not max_actions:
                # If we're given a set of allowed actions, and we're not just keeping the top k of
                # them, we don't need to do any sorting, so we can speed things up quite a bit.
                for group_index, log_probs, action_embeddings, actions in results:
                    for log_prob, action_embedding, action in zip(log_probs, action_embeddings, actions):
                        if action in allowed_actions[group_index]:
                            new_states.append(make_state(group_index, action, log_prob, action_embedding))
            else:
                # In this case, we need to sort the actions.  We'll do that on CPU, as it's easier,
                # and our action list is on the CPU, anyway.
                group_indices = []
                group_log_probs: List[torch.Tensor] = []
                group_action_embeddings = []
                group_actions = []
                for group_index, log_probs, action_embeddings, actions in results:
                    group_indices.extend([group_index] * len(actions))
                    group_log_probs.append(log_probs)
                    group_action_embeddings.append(action_embeddings)
                    group_actions.extend(actions)
                log_probs = torch.cat(group_log_probs, dim=0)
                action_embeddings = torch.cat(group_action_embeddings, dim=0)
                log_probs_cpu = log_probs.data.cpu().numpy().tolist()
                batch_states = [(log_probs_cpu[i],
                                 group_indices[i],
                                 log_probs[i],
                                 action_embeddings[i],
                                 group_actions[i])
                                for i in range(len(group_actions))
                                if (not allowed_actions or
                                    group_actions[i] in allowed_actions[group_indices[i]])]
                # We use a key here to make sure we're not trying to compare anything on the GPU.
                batch_states.sort(key=lambda x: x[0], reverse=True)
                if max_actions:
                    batch_states = batch_states[:max_actions]
                for _, group_index, log_prob, action_embedding, action in batch_states:
                    new_states.append(make_state(group_index, action, log_prob, action_embedding))
        return new_states

    def _take_first_step(self,
                         state: GrammarBasedState,
                         allowed_actions: List[Set[int]] = None) -> List[GrammarBasedState]:
        # We'll just do a projection from the current hidden state (which was initialized with the
        # final encoder output) to the number of start actions that we have, normalize those
        # logits, and use that as our score.  We end up duplicating some of the logic from
        # `_compute_new_states` here, but we do things slightly differently, and it's easier to
        # just copy the parts we need than to try to re-use that code.

        # (group_size, hidden_dim)
        hidden_state = torch.stack([rnn_state.hidden_state for rnn_state in state.rnn_state])
        # (group_size, num_start_type)
        start_action_logits = self._start_type_predictor(hidden_state)
        log_probs = torch.nn.functional.log_softmax(start_action_logits, dim=-1)
        sorted_log_probs, sorted_actions = log_probs.sort(dim=-1, descending=True)

        sorted_actions = sorted_actions.detach().cpu().numpy().tolist()
        if state.debug_info is not None:
            probs_cpu = log_probs.exp().detach().cpu().numpy().tolist()
        else:
            probs_cpu = [None] * len(state.batch_indices)

        # state.get_valid_actions() will return a list that is consistently sorted, so as along as
        # the set of valid start actions never changes, we can just match up the log prob indices
        # above with the position of each considered action, and we're good.
        valid_actions = state.get_valid_actions()
        considered_actions = [actions['global'][2] for actions in valid_actions]
        if len(considered_actions[0]) != self._num_start_types:
            raise RuntimeError("Calculated wrong number of initial actions.  Expected "
                               f"{self._num_start_types}, found {len(considered_actions[0])}.")

        best_next_states: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        for group_index, (batch_index, group_actions) in enumerate(zip(state.batch_indices, sorted_actions)):
            for action_index, action in enumerate(group_actions):
                # `action` is currently the index in `log_probs`, not the actual action ID.  To get
                # the action ID, we need to go through `considered_actions`.
                action = considered_actions[group_index][action]
                if allowed_actions is not None and action not in allowed_actions[group_index]:
                    # This happens when our _decoder trainer_ wants us to only evaluate certain
                    # actions, likely because they are the gold actions in this state.  We just skip
                    # emitting any state that isn't allowed by the trainer, because constructing the
                    # new state can be expensive.
                    continue
                best_next_states[batch_index].append((group_index, action_index, action))

        new_states = []
        for batch_index, best_states in sorted(best_next_states.items()):
            for group_index, action_index, action in best_states:
                # We'll yield a bunch of states here that all have a `group_size` of 1, so that the
                # learning algorithm can decide how many of these it wants to keep, and it can just
                # regroup them later, as that's a really easy operation.
                new_score = state.score[group_index] + sorted_log_probs[group_index, action_index]

                # This part is different from `_compute_new_states` - we're just passing through
                # the previous RNN state, as predicting the start type wasn't included in the
                # decoder RNN in the original model.
                new_rnn_state = state.rnn_state[group_index]
                new_state = state.new_state_from_group_index(group_index,
                                                             action,
                                                             new_score,
                                                             new_rnn_state,
                                                             considered_actions[group_index],
                                                             probs_cpu[group_index],
                                                             None)
                new_states.append(new_state)
        return new_states

    def attend_on_question(self,
                           query: torch.Tensor,
                           encoder_outputs: torch.Tensor,
                           encoder_output_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a query (which is typically the decoder hidden state), compute an attention over the
        output of the question encoder, and return a weighted sum of the question representations
        given this attention.  We also return the attention weights themselves.

        This is a simple computation, but we have it as a separate method so that the ``forward``
        method on the main parser module can call it on the initial hidden state, to simplify the
        logic in ``take_step``.
        """
        # (group_size, question_length)
        question_attention_weights = self._input_attention(query,
                                                           encoder_outputs,
                                                           encoder_output_mask)
        # (group_size, encoder_output_dim)
        attended_question = util.weighted_sum(encoder_outputs, question_attention_weights)
        return attended_question, question_attention_weights
