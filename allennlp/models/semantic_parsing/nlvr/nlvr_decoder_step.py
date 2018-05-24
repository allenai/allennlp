from typing import List, Dict, Tuple, Set
from collections import defaultdict

from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear

from allennlp.common import util as common_util
from allennlp.models.semantic_parsing.nlvr.nlvr_decoder_state import NlvrDecoderState
from allennlp.modules import Attention
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.decoding import DecoderStep, RnnState, ChecklistState
from allennlp.nn import util as nn_util


class NlvrDecoderStep(DecoderStep[NlvrDecoderState]):
    """
    Parameters
    ----------
    encoder_output_dim : ``int``
    action_embedding_dim : ``int``
    attention_function : ``SimilarityFunction``
    dropout : ``float``
        Dropout to use on decoder outputs and before action prediction.
    use_coverage : ``bool``, optional (default=False)
        Is this DecoderStep being used in a semantic parser trained using coverage? We need to know
        this to define a learned parameter for using checklist balances in action prediction.
    """
    def __init__(self,
                 encoder_output_dim: int,
                 action_embedding_dim: int,
                 attention_function: SimilarityFunction,
                 dropout: float = 0.0,
                 use_coverage: bool = False) -> None:
        super(NlvrDecoderStep, self).__init__()
        self._input_attention = Attention(attention_function)

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
        if use_coverage:
            # This is a multiplicative factor that is used to add the embeddings of yet to be
            # produced actions to the predicted embedding and bias it.
            self._checklist_embedding_multiplier = Parameter(torch.FloatTensor([1.0]))
        # TODO(pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(input_dim, output_dim)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    @overrides
    def take_step(self,  # type: ignore
                  state: NlvrDecoderState,
                  max_actions: int = None,
                  allowed_actions: List[Set[int]] = None) -> List[NlvrDecoderState]:
        """
        Given a ``NlvrDecoderState``, returns a list of next states that are sorted by their scores.
        This method is very similar to ``WikiTablesDecoderStep._take_step``. The differences are
        that depending on the type of supervision being used, we may not have a notion of
        "allowed actions" here, and we do not perform entity linking here.
        """
        # Outline here: first we'll construct the input to the decoder, which is a concatenation of
        # an embedding of the decoder input (the last action taken) and an attention over the
        # sentence.  Then we'll update our decoder's hidden state given this input, and recompute
        # an attention over the sentence given our new hidden state.  We'll use a concatenation of
        # the new hidden state, the new attention, and optionally the checklist balance to predict an
        # output, then yield new states. We will compute and use a checklist balance when
        # ``allowed_actions`` is None, with the assumption that the ``DecoderTrainer`` that is
        # calling this method is trying to train a parser without logical form supervision.
        # TODO (pradeep): Make the distinction between the two kinds of trainers in the way they
        # call this method more explicit.

        # Each new state corresponds to one valid action that can be taken from the current state,
        # and they are ordered by model scores.
        attended_sentence = torch.stack([rnn_state.attended_input for rnn_state in state.rnn_state])
        hidden_state = torch.stack([rnn_state.hidden_state for rnn_state in state.rnn_state])
        memory_cell = torch.stack([rnn_state.memory_cell for rnn_state in state.rnn_state])
        previous_action_embedding = torch.stack([rnn_state.previous_action_embedding
                                                 for rnn_state in state.rnn_state])

        # (group_size, decoder_input_dim)
        decoder_input = self._input_projection_layer(torch.cat([attended_sentence,
                                                                previous_action_embedding], -1))
        decoder_input = torch.nn.functional.tanh(decoder_input)
        hidden_state, memory_cell = self._decoder_cell(decoder_input, (hidden_state, memory_cell))

        hidden_state = self._dropout(hidden_state)
        # (group_size, encoder_output_dim)
        encoder_outputs = torch.stack([state.rnn_state[0].encoder_outputs[i] for i in state.batch_indices])
        encoder_output_mask = torch.stack([state.rnn_state[0].encoder_output_mask[i] for i in state.batch_indices])
        attended_sentence = self.attend_on_sentence(hidden_state, encoder_outputs, encoder_output_mask)

        # We get global indices of actions to embed here. The following logic is similar to
        # ``WikiTablesDecoderStep._get_actions_to_consider``, except that we do not have any actions
        # to link.
        valid_actions = state.get_valid_actions()
        global_valid_actions: List[List[Tuple[int, int]]] = []
        for batch_index, valid_action_list in zip(state.batch_indices, valid_actions):
            global_valid_actions.append([])
            for action_index in valid_action_list:
                # state.action_indices is a dictionary that maps (batch_index, batch_action_index)
                # to global_action_index
                global_action_index = state.action_indices[(batch_index, action_index)]
                global_valid_actions[-1].append((global_action_index, action_index))
        global_actions_to_embed: List[List[int]] = []
        local_actions: List[List[int]] = []
        for global_action_list in global_valid_actions:
            global_action_list.sort()
            global_actions_to_embed.append([])
            local_actions.append([])
            for global_action_index, action_index in global_action_list:
                global_actions_to_embed[-1].append(global_action_index)
                local_actions[-1].append(action_index)
        max_num_actions = max([len(action_list) for action_list in global_actions_to_embed])
        # We pad local actions with -1 as padding to get considered actions.
        considered_actions = [common_util.pad_sequence_to_length(action_list, max_num_actions,
                                                                 default_value=lambda: -1)
                              for action_list in local_actions]

        # action_embeddings: (group_size, num_embedded_actions, action_embedding_dim)
        # action_mask: (group_size, num_embedded_actions)
        action_embeddings, embedded_action_mask = self._get_action_embeddings(state,
                                                                              global_actions_to_embed)
        action_query = torch.cat([hidden_state, attended_sentence], dim=-1)
        # (group_size, action_embedding_dim)
        predicted_action_embedding = self._output_projection_layer(action_query)
        predicted_action_embedding = self._dropout(torch.nn.functional.tanh(predicted_action_embedding))
        if state.checklist_state[0] is not None:
            embedding_addition = self._get_predicted_embedding_addition(state)
            addition = embedding_addition * self._checklist_embedding_multiplier
            predicted_action_embedding = predicted_action_embedding + addition
        # We'll do a batch dot product here with `bmm`.  We want `dot(predicted_action_embedding,
        # action_embedding)` for each `action_embedding`, and we can get that efficiently with
        # `bmm` and some squeezing.
        # Shape: (group_size, num_embedded_actions)
        action_logits = action_embeddings.bmm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)

        action_mask = embedded_action_mask.float()
        if state.checklist_state[0] is not None:
            # We will compute the logprobs and the checklists of potential next states together for
            # efficiency.
            logprobs, new_checklist_states = self._get_next_state_info_with_agenda(state,
                                                                                   considered_actions,
                                                                                   action_logits,
                                                                                   action_mask)
        else:
            logprobs = self._get_next_state_info_without_agenda(state,
                                                                considered_actions,
                                                                action_logits,
                                                                action_mask)
            new_checklist_states = None
        return self._compute_new_states(state,
                                        logprobs,
                                        hidden_state,
                                        memory_cell,
                                        action_embeddings,
                                        attended_sentence,
                                        considered_actions,
                                        allowed_actions,
                                        new_checklist_states,
                                        max_actions)

    @staticmethod
    def _get_predicted_embedding_addition(state: NlvrDecoderState) -> torch.Tensor:
        """
        Computes checklist balance, uses it to get the embeddings of desired terminal actions yet to
        be produced by the decoder, and returns their sum for the decoder to add it to the predicted
        embedding to bias the prediction towards missing actions.
        """
        # This holds a list of checklist balances for this state. Each balance is a float vector
        # containing just 1s and 0s showing which of the items are filled. We clamp the min at 0
        # to ignore the number of times an action is taken. The value at an index will be 1 iff
        # the target wants an unmasked action to be taken, and it is not yet taken. All elements
        # in each balance corresponding to masked actions will be 0.
        checklist_balances = []
        for instance_checklist_state in state.checklist_state:
            checklist_balances.append(instance_checklist_state.get_balance())

        # Note that the checklist balance has 0s for actions that are masked (see comment above).
        # So, masked actions do not contribute to the projection of ``predicted_action_emebedding``
        # below.
        # (group_size, num_terminals, 1)
        checklist_balance = torch.stack([x for x in  checklist_balances])
        global_terminal_indices: List[List[int]] = []
        # We map the group indices of all terminal actions that are relevant to checklist
        # computation, to global indices here.
        for batch_index, checklist_state in zip(state.batch_indices, state.checklist_state):
            global_terminal_indices.append([])
            for terminal_index in checklist_state.terminal_actions.data.cpu():
                global_terminal_index = state.action_indices[(batch_index, int(terminal_index[0]))]
                global_terminal_indices[-1].append(global_terminal_index)
        # We don't need to pad this tensor because the terminal indices from all groups will be the
        # same size.
        terminal_indices_tensor = Variable(state.score[0].data.new(global_terminal_indices)).long()
        group_size = len(state.batch_indices)
        action_embedding_dim = state.action_embeddings.size(-1)
        num_terminals = len(global_terminal_indices[0])
        flattened_terminal_indices = terminal_indices_tensor.view(-1)
        flattened_action_embeddings = state.action_embeddings.index_select(0,
                                                                           flattened_terminal_indices)
        terminal_embeddings = flattened_action_embeddings.view(group_size, num_terminals, action_embedding_dim)
        checklist_balance_embeddings = terminal_embeddings * checklist_balance
        # (group_size, action_embedding_dim)
        return checklist_balance_embeddings.sum(1)

    @staticmethod
    def _get_next_state_info_with_agenda(
            state: NlvrDecoderState,
            considered_actions: List[List[int]],
            action_logits: torch.Tensor,
            action_mask: torch.Tensor) -> Tuple[List[List[Tuple[int, torch.LongTensor]]],
                                                List[List[ChecklistState]]]:
        """
        We return a list of log probabilities and checklist states corresponding to next actions that are
        not padding. This method is applicable to the case where we do not have target action
        sequences and are relying on agendas for training.
        """
        considered_action_probs = nn_util.masked_softmax(action_logits, action_mask)
        # Mixing model scores and agenda selection probabilities to compute the probabilities of all
        # actions for the next step and the corresponding new checklists.
        # All action logprobs will keep track of logprob corresponding to each local action index
        # for each instance.
        all_action_logprobs: List[List[Tuple[int, torch.LongTensor]]] = []
        all_new_checklist_states: List[List[ChecklistState]] = []
        for group_index, instance_info in enumerate(zip(state.score,
                                                        considered_action_probs,
                                                        state.checklist_state)):
            (instance_score, instance_probs, instance_checklist_state) = instance_info
            # We will mix the model scores with agenda selection probabilities and compute their
            # logs to fill the following list with action indices and corresponding logprobs.
            instance_action_logprobs: List[Tuple[int, torch.Tensor]] = []
            instance_new_checklist_states: List[ChecklistState] = []
            for action_index, action_prob in enumerate(instance_probs):
                # This is the actual index of the action from the original list of actions.
                action = considered_actions[group_index][action_index]
                if action == -1:
                    # Ignoring padding.
                    continue
                new_checklist_state = instance_checklist_state.update(action)  # (terminal_actions, 1)
                instance_new_checklist_states.append(new_checklist_state)
                logprob = instance_score + torch.log(action_prob + 1e-13)
                instance_action_logprobs.append((action_index, logprob))
            all_action_logprobs.append(instance_action_logprobs)
            all_new_checklist_states.append(instance_new_checklist_states)
        return all_action_logprobs, all_new_checklist_states

    @staticmethod
    def _get_next_state_info_without_agenda(state: NlvrDecoderState,
                                            considered_actions: List[List[int]],
                                            action_logits: torch.Tensor,
                                            action_mask: torch.Tensor) -> List[List[Tuple[int,
                                                                                          torch.LongTensor]]]:
        """
        We return a list of log probabilities corresponding to actions that are not padding. This
        method is related to the training scenario where we have target action sequences for
        training.
        """
        considered_action_logprobs = nn_util.masked_log_softmax(action_logits, action_mask)
        all_action_logprobs: List[List[Tuple[int, torch.LongTensor]]] = []
        for group_index, (score, considered_logprobs) in enumerate(zip(state.score,
                                                                       considered_action_logprobs)):
            instance_action_logprobs: List[Tuple[int, torch.Tensor]] = []
            for action_index, logprob in enumerate(considered_logprobs):
                # This is the actual index of the action from the original list of actions.
                action = considered_actions[group_index][action_index]
                if action == -1:
                    # Ignoring padding.
                    continue
                instance_action_logprobs.append((action_index, score + logprob))
            all_action_logprobs.append(instance_action_logprobs)
        return all_action_logprobs

    def attend_on_sentence(self,
                           query: torch.Tensor,
                           encoder_outputs: torch.Tensor,
                           encoder_output_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method is almost identical to ``WikiTablesDecoderStep.attend_on_question``. We just
        don't return the attention weights.
        Given a query (which is typically the decoder hidden state), compute an attention over the
        output of the sentence encoder, and return a weighted sum of the sentence representations
        given this attention.  We also return the attention weights themselves.

        This is a simple computation, but we have it as a separate method so that the ``forward``
        method on the main parser module can call it on the initial hidden state, to simplify the
        logic in ``take_step``.
        """
        # (group_size, sentence_length)
        sentence_attention_weights = self._input_attention(query,
                                                           encoder_outputs,
                                                           encoder_output_mask)
        # (group_size, encoder_output_dim)
        attended_sentence = nn_util.weighted_sum(encoder_outputs, sentence_attention_weights)
        return attended_sentence

    @staticmethod
    def _get_action_embeddings(state: NlvrDecoderState,
                               actions_to_embed: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method is identical to ``WikiTablesDecoderStep._get_action_embeddings``
        Returns an embedded representation for all actions in ``actions_to_embed``, using the state
        in ``NlvrDecoderState``.

        Parameters
        ----------
        state : ``NlvrDecoderState``
            The current state.  We'll use this to get the global action embeddings.
        actions_to_embed : ``List[List[int]]``
            A list of _global_ action indices for each group element.  Should have shape
            (group_size, num_actions), unpadded.

        Returns
        -------
        action_embeddings : ``torch.FloatTensor``
            An embedded representation of all of the given actions.  Shape is ``(group_size,
            num_actions, action_embedding_dim)``, where ``num_actions`` is the maximum number of
            considered actions for any group element.
        action_mask : ``torch.LongTensor``
            A mask of shape ``(group_size, num_actions)`` indicating which ``(group_index,
            action_index)`` pairs were merely added as padding.
        """
        num_actions = [len(action_list) for action_list in actions_to_embed]
        max_num_actions = max(num_actions)
        padded_actions = [common_util.pad_sequence_to_length(action_list, max_num_actions)
                          for action_list in actions_to_embed]
        # Shape: (group_size, num_actions)
        action_tensor = Variable(state.score[0].data.new(padded_actions).long())
        # `state.action_embeddings` is shape (total_num_actions, action_embedding_dim).
        # We want to select from state.action_embeddings using `action_tensor` to get a tensor of
        # shape (group_size, num_actions, action_embedding_dim).  Unfortunately, the index_select
        # functions in nn.util don't do this operation.  So we'll do some reshapes and do the
        # index_select ourselves.
        group_size = len(state.batch_indices)
        action_embedding_dim = state.action_embeddings.size(-1)
        flattened_actions = action_tensor.view(-1)
        flattened_action_embeddings = state.action_embeddings.index_select(0, flattened_actions)
        action_embeddings = flattened_action_embeddings.view(group_size, max_num_actions, action_embedding_dim)
        sequence_lengths = Variable(action_embeddings.data.new(num_actions))
        action_mask = nn_util.get_mask_from_sequence_lengths(sequence_lengths, max_num_actions)
        return action_embeddings, action_mask

    @classmethod
    def _compute_new_states(cls,
                            state: NlvrDecoderState,
                            action_logprobs: List[List[Tuple[int, torch.Tensor]]],
                            hidden_state: torch.Tensor,
                            memory_cell: torch.Tensor,
                            action_embeddings: torch.Tensor,
                            attended_sentence: torch.Tensor,
                            considered_actions: List[List[int]],
                            allowed_actions: List[Set[int]] = None,
                            new_checklist_states: List[List[ChecklistState]] = None,
                            max_actions: int = None) -> List[NlvrDecoderState]:
        """
        This method is very similar to ``WikiTabledDecoderStep._compute_new_states``.
        The difference here is that we also keep track of checklists if they are passed to this
        method.
        """
        # batch_index -> group_index, action_index, checklist, score
        states_info: Dict[int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        if new_checklist_states is None:
            # We do not have checklist states. Making a list of lists of Nones of the appropriate size for
            # the zips below.
            new_checklist_states = [[None for logprob in instance_logprobs]
                                    for instance_logprobs in action_logprobs]
        for group_index, instance_info in enumerate(zip(state.batch_indices,
                                                        action_logprobs,
                                                        new_checklist_states)):
            batch_index, instance_logprobs, instance_new_checklist_states = instance_info
            for (action_index, score), checklist_state in zip(instance_logprobs,
                                                              instance_new_checklist_states):
                states_info[batch_index].append((group_index, action_index, checklist_state, score))

        new_states = []
        for batch_index, instance_states_info in states_info.items():
            batch_scores = torch.cat([info[-1] for info in instance_states_info])
            _, sorted_indices = batch_scores.sort(-1, descending=True)
            sorted_states_info = [instance_states_info[i] for i in sorted_indices.data.cpu().numpy()]
            allowed_states_info = []
            for i, (group_index, action_index, _, _) in enumerate(sorted_states_info):
                action = considered_actions[group_index][action_index]
                if allowed_actions is not None and action not in allowed_actions[group_index]:
                    continue
                allowed_states_info.append(sorted_states_info[i])
            sorted_states_info = allowed_states_info
            if max_actions is not None:
                sorted_states_info = sorted_states_info[:max_actions]
            for group_index, action_index, new_checklist_state, new_score in sorted_states_info:
                # This is the actual index of the action from the original list of actions.
                # We do not have to check whether it is the padding index because ``take_step``
                # already took care of that.
                action = considered_actions[group_index][action_index]
                action_embedding = action_embeddings[group_index, action_index, :]
                new_action_history = state.action_history[group_index] + [action]
                production_rule = state.possible_actions[batch_index][action][0]
                new_grammar_state = state.grammar_state[group_index].take_action(production_rule)
                new_rnn_state = RnnState(hidden_state[group_index],
                                         memory_cell[group_index],
                                         action_embedding,
                                         attended_sentence[group_index],
                                         state.rnn_state[group_index].encoder_outputs,
                                         state.rnn_state[group_index].encoder_output_mask)
                new_state = NlvrDecoderState(batch_indices=[batch_index],
                                             action_history=[new_action_history],
                                             score=[new_score],
                                             rnn_state=[new_rnn_state],
                                             grammar_state=[new_grammar_state],
                                             action_embeddings=state.action_embeddings,
                                             action_indices=state.action_indices,
                                             possible_actions=state.possible_actions,
                                             worlds=state.worlds,
                                             label_strings=state.label_strings,
                                             checklist_state=[new_checklist_state])
                new_states.append(new_state)
        return new_states
