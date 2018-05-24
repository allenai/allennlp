from collections import defaultdict
from typing import Dict, List, Set, Tuple

from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear

from allennlp.common import util as common_util
from allennlp.common.checks import check_dimensions_match
from allennlp.models.semantic_parsing.wikitables.wikitables_decoder_state import WikiTablesDecoderState
from allennlp.modules import Attention, FeedForward
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.decoding import DecoderStep, RnnState


class WikiTablesDecoderStep(DecoderStep[WikiTablesDecoderState]):
    """
    Parameters
    ----------
    encoder_output_dim : ``int``
    action_embedding_dim : ``int``
    attention_function : ``SimilarityFunction``
    num_start_types : ``int``
    num_entity_types : ``int``
    mixture_feedforward : ``FeedForward`` (optional, default=None)
    dropout : ``float`` (optional, default=0.0)
    unlinked_terminal_indices : ``List[int]``, (optional, default=None)
        If we are training a parser to maximize coverage using a checklist, we need to know the
        global indices of the unlinked terminal productions to be able to compute the checklist
        corresponding to those terminals, and project a concatenation of the current hidden
        state, attended encoder input and the current checklist balance into the action space.
        This is not needed if we are training the parser using target action sequences.
    """
    def __init__(self,
                 encoder_output_dim: int,
                 action_embedding_dim: int,
                 attention_function: SimilarityFunction,
                 num_start_types: int,
                 num_entity_types: int,
                 mixture_feedforward: FeedForward = None,
                 dropout: float = 0.0,
                 unlinked_terminal_indices: List[int] = None) -> None:
        super(WikiTablesDecoderStep, self).__init__()
        self._mixture_feedforward = mixture_feedforward
        self._entity_type_embedding = Embedding(num_entity_types, action_embedding_dim)
        self._input_attention = Attention(attention_function)

        self._num_start_types = num_start_types
        self._start_type_predictor = Linear(encoder_output_dim, num_start_types)

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
        if unlinked_terminal_indices is not None:
            # This means we are using coverage to train the parser.
            # These factors are used to add the embeddings of yet to be produced actions to the
            # predicted embedding, and to boost the action logits of yet to be produced linked
            # actions, respectively.
            self._unlinked_checklist_multiplier = Parameter(torch.FloatTensor([1.0]))
            self._linked_checklist_multiplier = Parameter(torch.FloatTensor([1.0]))

        self._unlinked_terminal_indices = unlinked_terminal_indices
        # TODO(pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(input_dim, output_dim)

        if mixture_feedforward is not None:
            check_dimensions_match(output_dim, mixture_feedforward.get_input_dim(),
                                   "hidden state embedding dim", "mixture feedforward input dim")
            check_dimensions_match(mixture_feedforward.get_output_dim(), 1,
                                   "mixture feedforward output dim", "dimension for scalar value")

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    @overrides
    def take_step(self,
                  state: WikiTablesDecoderState,
                  max_actions: int = None,
                  allowed_actions: List[Set[int]] = None) -> List[WikiTablesDecoderState]:
        if not state.action_history[0]:
            # The wikitables parser did something different when predicting the start type, which
            # is our first action.  So in this case we break out into a different function.  We'll
            # ignore max_actions on our first step, assuming there aren't that many start types.
            return self._take_first_step(state, allowed_actions)
        # Outline here: first we'll construct the input to the decoder, which is a concatenation of
        # an embedding of the decoder input (the last action taken) and an attention over the
        # question.  Then we'll update our decoder's hidden state given this input, and recompute an
        # attention over the question given our new hidden state.  We'll use a concatenation of the
        # new hidden state and the new attention, and optionally the checklist balance, to predict an
        # output, then yield new states.  Each new state corresponds to one valid action that can be
        # taken from the current state, and they are ordered by their probability of being selected.
        attended_question = torch.stack([rnn_state.attended_input for rnn_state in state.rnn_state])
        hidden_state = torch.stack([rnn_state.hidden_state for rnn_state in state.rnn_state])
        memory_cell = torch.stack([rnn_state.memory_cell for rnn_state in state.rnn_state])
        previous_action_embedding = torch.stack([rnn_state.previous_action_embedding
                                                 for rnn_state in state.rnn_state])

        # The scores from all prior state transitions until now.  Shape: (group_size, 1).
        scores_so_far = torch.stack(state.score)

        # (group_size, decoder_input_dim)
        projected_input = self._input_projection_layer(torch.cat([attended_question,
                                                                  previous_action_embedding], -1))
        decoder_input = torch.nn.functional.relu(projected_input)

        hidden_state, memory_cell = self._decoder_cell(decoder_input, (hidden_state, memory_cell))
        hidden_state = self._dropout(hidden_state)

        # (group_size, encoder_output_dim)
        encoder_outputs = torch.stack([state.rnn_state[0].encoder_outputs[i] for i in state.batch_indices])
        encoder_output_mask = torch.stack([state.rnn_state[0].encoder_output_mask[i] for i in state.batch_indices])
        attended_question, attention_weights = self.attend_on_question(hidden_state,
                                                                       encoder_outputs,
                                                                       encoder_output_mask)
        action_query = torch.cat([hidden_state, attended_question], dim=-1)
        considered_actions, actions_to_embed, actions_to_link = self._get_actions_to_consider(state)
        # action_embeddings: (group_size, num_embedded_actions, action_embedding_dim)
        # output_action_embeddings: (group_size, num_embedded_actions, action_embedding_dim)
        # action_mask: (group_size, num_embedded_actions)
        action_embeddings, output_action_embeddings, action_biases, embedded_action_mask = \
                self._get_action_embeddings(state, actions_to_embed)
        # (group_size, action_embedding_dim)
        projected_query = torch.nn.functional.relu(self._output_projection_layer(action_query))
        predicted_action_embedding = self._dropout(projected_query)
        linked_balance = None
        if state.checklist_state[0] is not None:
            linked_balance, unlinked_balance = self._get_checklist_balance(state,
                                                                           self._unlinked_terminal_indices,
                                                                           actions_to_link)
            embedding_addition = self._get_predicted_embedding_addition(state,
                                                                        self._unlinked_terminal_indices,
                                                                        unlinked_balance)
            addition = embedding_addition * self._unlinked_checklist_multiplier
            predicted_action_embedding = predicted_action_embedding + addition

        # We'll do a batch dot product here with `bmm`.  We want `dot(predicted_action_embedding,
        # action_embedding)` for each `action_embedding`, and we can get that efficiently with
        # `bmm` and some squeezing.
        # Shape: (group_size, num_embedded_actions)
        embedded_action_logits = action_embeddings.bmm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
        embedded_action_logits = embedded_action_logits + action_biases.squeeze(-1)
        if actions_to_link:
            # entity_action_logits: (group_size, num_entity_actions)
            # entity_action_mask: (group_size, num_entity_actions)
            entity_action_logits, entity_action_mask, entity_type_embeddings = \
                    self._get_entity_action_logits(state,
                                                   actions_to_link,
                                                   attention_weights,
                                                   linked_balance)

            # The `output_action_embeddings` tensor gets used later as the input to the next
            # decoder step.  For linked actions, we don't have any action embedding, so we use the
            # entity type instead.
            output_action_embeddings = torch.cat([output_action_embeddings, entity_type_embeddings], dim=1)

            if self._mixture_feedforward is not None:
                # The entity and action logits are combined with a mixture weight to prevent the
                # entity_action_logits from dominating the embedded_action_logits if a softmax
                # was applied on both together.
                mixture_weight = self._mixture_feedforward(hidden_state)
                mix1 = torch.log(mixture_weight)
                mix2 = torch.log(1 - mixture_weight)

                entity_action_probs = util.masked_log_softmax(entity_action_logits,
                                                              entity_action_mask.float()) + mix1
                embedded_action_probs = util.masked_log_softmax(embedded_action_logits,
                                                                embedded_action_mask.float()) + mix2
                current_log_probs = torch.cat([embedded_action_probs, entity_action_probs], dim=1)
            else:
                action_logits = torch.cat([embedded_action_logits, entity_action_logits], dim=1)
                action_mask = torch.cat([embedded_action_mask, entity_action_mask], dim=1).float()
                current_log_probs = util.masked_log_softmax(action_logits, action_mask)
        else:
            action_logits = embedded_action_logits
            action_mask = embedded_action_mask.float()
            current_log_probs = util.masked_log_softmax(action_logits, action_mask)

        # current_log_probs is shape (group_size, num_actions).  We're broadcasting an addition
        # here with scores_so_far, which has shape (group_size, 1).  This is now the total score
        # for each state after taking each action.  We're going to sort by this in
        # `_compute_new_states`, so it's important that this is the total score, not just the score
        # for the current action.
        log_probs = scores_so_far + current_log_probs

        return self._compute_new_states(state,
                                        log_probs,
                                        hidden_state,
                                        memory_cell,
                                        output_action_embeddings,
                                        attended_question,
                                        attention_weights,
                                        considered_actions,
                                        allowed_actions,
                                        max_actions)

    @staticmethod
    def _get_checklist_balance(state: WikiTablesDecoderState,
                               unlinked_terminal_indices: List[int],
                               actions_to_link: List[List[int]]) -> Tuple[torch.FloatTensor,
                                                                          torch.FloatTensor]:
        # This holds a list of checklist balances for this state. Each balance is a float vector
        # containing just 1s and 0s showing which of the items are filled. We clamp the min at 0
        # to ignore the number of times an action is taken. The value at an index will be 1 iff
        # the target wants an unmasked action to be taken, and it is not yet taken. All elements
        # in each balance corresponding to masked actions will be 0.
        checklist_balances = []
        for instance_checklist_state in state.checklist_state:
            checklist_balance = torch.clamp(instance_checklist_state.get_balance(), min=0.0)
            checklist_balances.append(checklist_balance)

        checklist_balance = torch.stack([x for x in  checklist_balances])
        checklist_balance = checklist_balance.squeeze(2)  # (group_size, num_terminals)
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
        unlinked_action_indices = Variable(checklist_balance.data.new(mapped_actions_to_embed)).long()
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
            linked_action_indices = Variable(checklist_balance.data.new(padded_actions_to_link)).long()
            linked_actions_mask = (linked_action_indices != -1).long()
            linked_action_indices = linked_action_indices * linked_actions_mask
            linked_checklist_balance = torch.gather(checklist_balance, 1, linked_action_indices)
            linked_checklist_balance = linked_checklist_balance * linked_actions_mask.float()
        return linked_checklist_balance, unlinked_checklist_balance

    @staticmethod
    def _get_predicted_embedding_addition(state: WikiTablesDecoderState,
                                          unlinked_terminal_indices: List[int],
                                          unlinked_checklist_balance: torch.Tensor) -> torch.Tensor:
        """
        Gets the embeddings of desired unlinked terminal actions yet to be produced by the decoder,
        and returns their sum for the decoder to add it to the predicted embedding to bias the
        prediction towards missing actions.
        """
        # (group_size, num_unlinked_actions, 1)
        unlinked_balance = unlinked_checklist_balance.unsqueeze(2)
        group_size = len(state.batch_indices)
        action_embedding_dim = state.action_embeddings.size(-1)
        num_terminals = len(unlinked_terminal_indices)
        group_terminal_indices = [unlinked_terminal_indices for _ in range(group_size)]
        # (group_size, num_unlinked_actions)
        terminal_indices_tensor = Variable(state.score[0].data.new(group_terminal_indices)).long()
        flattened_terminal_indices = terminal_indices_tensor.view(-1)
        flattened_action_embeddings = state.action_embeddings.index_select(0,
                                                                           flattened_terminal_indices)
        terminal_embeddings = flattened_action_embeddings.view(group_size, num_terminals, action_embedding_dim)
        checklist_balance_embeddings = terminal_embeddings * unlinked_balance
        # (group_size, action_embedding_dim)
        return checklist_balance_embeddings.sum(1)

    def _take_first_step(self,
                         state: WikiTablesDecoderState,
                         allowed_actions: List[Set[int]] = None) -> List[WikiTablesDecoderState]:
        # We'll just do a projection from the current hidden state (which was initialized with the
        # final encoder output) to the number of start actions that we have, normalize those
        # logits, and use that as our score.  We end up duplicating some of the logic from
        # `_compute_new_states` here, but we do things slightly differently, and it's easier to
        # just copy the parts we need than to try to re-use that code.

        # (group_size, hidden_dim)
        hidden_state = torch.stack([rnn_state.hidden_state for rnn_state in state.rnn_state])
        # (group_size, num_start_type)
        start_action_logits = self._start_type_predictor(hidden_state)
        log_probs = util.masked_log_softmax(start_action_logits, None)
        sorted_log_probs, sorted_actions = log_probs.sort(dim=-1, descending=True)

        sorted_actions = sorted_actions.data.cpu().numpy().tolist()
        if state.debug_info is not None:
            probs_cpu = log_probs.exp().data.cpu().numpy().tolist()

        # state.get_valid_actions() will return a list that is consistently sorted, so as along as
        # the set of valid start actions never changes, we can just match up the log prob indices
        # above with the position of each considered action, and we're good.
        considered_actions, _, _ = self._get_actions_to_consider(state)
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
                batch_index = state.batch_indices[group_index]
                new_action_history = state.action_history[group_index] + [action]
                new_score = state.score[group_index] + sorted_log_probs[group_index, action_index]

                production_rule = state.possible_actions[batch_index][action][0]
                new_grammar_state = state.grammar_state[group_index].take_action(production_rule)
                new_checklist_state = [state.checklist_state[group_index]]
                if state.debug_info is not None:
                    debug_info = {
                            'considered_actions': considered_actions[group_index],
                            'probabilities': probs_cpu[group_index],
                            }
                    new_debug_info = [state.debug_info[group_index] + [debug_info]]
                else:
                    new_debug_info = None

                # This part is different from `_compute_new_states` - we're just passing through
                # the previous RNN state, as predicting the start type wasn't included in the
                # decoder RNN in the original model.
                new_rnn_state = state.rnn_state[group_index]
                new_state = WikiTablesDecoderState(batch_indices=[batch_index],
                                                   action_history=[new_action_history],
                                                   score=[new_score],
                                                   rnn_state=[new_rnn_state],
                                                   grammar_state=[new_grammar_state],
                                                   action_embeddings=state.action_embeddings,
                                                   output_action_embeddings=state.output_action_embeddings,
                                                   action_biases=state.action_biases,
                                                   action_indices=state.action_indices,
                                                   possible_actions=state.possible_actions,
                                                   flattened_linking_scores=state.flattened_linking_scores,
                                                   actions_to_entities=state.actions_to_entities,
                                                   entity_types=state.entity_types,
                                                   world=state.world,
                                                   example_lisp_string=state.example_lisp_string,
                                                   checklist_state=new_checklist_state,
                                                   debug_info=new_debug_info)
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

    @staticmethod
    def _get_actions_to_consider(state: WikiTablesDecoderState) -> Tuple[List[List[int]],
                                                                         List[List[int]],
                                                                         List[List[int]]]:
        """
        The ``WikiTablesDecoderState`` defines a set of actions that are valid in the current
        grammar state for each group element.  This method gets that set of actions and separates
        them into actions that can be embedded and actions that need to be linked.

        This method goes through all of the actions from ``state.get_valid_actions()`` and
        separates them into actions that can be embedded and actions that need to be linked, based
        on the action's ``global_action_index`` (all embeddable actions have an action index lower
        than the number of global embeddable actions).  After separating the actions, we combine
        them again, getting a padded list of all considered actions that can be used by
        :func:`_compute_new_states`.  All three of these lists are returned (the embeddable
        actions, the actions that need to be linked, and the padded collection of all actions that
        were considered).

        Returns
        -------
        considered_actions : ``List[List[int]]``
            A sorted list of all actions considered for each group element, both for embedding and
            for linking.  This list has one inner list for each group element, and each item in the
            inner list represents ``batch_action_index`` that was considered.  This inner list is
            also `padded` to size ``max_num_embedded_actions + max_num_linked_actions``, with
            `interior` padding in between the embedded actions and the linked actions where
            necessary.  The ``action_index`` for padded entries is -1.  This padding replicates the
            structre that we'll get in the model once we concatenate logits together, so that
            :func:`_compute_new_states` has an easy time figuring out what to do.
        actions_to_embed : ``List[List[int]]``
            These actions are in the global action embedding tensor, and can be embedded.  Shape is
            (group_size, num_actions), not padded, and the value is the ``global_action_index``,
            not the ``batch_action_index``.  You can use these indices to ``index_select`` on the
            global action embeddings directly, without additional translation.
        actions_to_link : ``List[List[int]]``
            These actions are `not` in the global action embedding tensor, and must have scores
            computed some way other than with an embedding.  Shape is (group_size, num_actions),
            not padded, and the value is the ``batch_action_index``.  These need to be converted
            into batch entity indices, then looked up in the linking scores.

            If there are `no` actions to link, because all actions have an embedding, we return
            `None` here.
        """
        # A list of `batch_action_indices` for each group element.
        valid_actions = state.get_valid_actions()
        global_valid_actions: List[List[Tuple[int, int]]] = []
        for batch_index, valid_action_list in zip(state.batch_indices, valid_actions):
            global_valid_actions.append([])
            for action_index in valid_action_list:
                # state.action_indices is a dictionary that maps (batch_index, batch_action_index)
                # to global_action_index
                global_action_index = state.action_indices[(batch_index, action_index)]
                global_valid_actions[-1].append((global_action_index, action_index))
        embedded_actions: List[List[int]] = []
        linked_actions: List[List[int]] = []
        for global_action_list in global_valid_actions:
            embedded_actions.append([])
            linked_actions.append([])
            for global_action_index, action_index in global_action_list:
                if global_action_index == -1:
                    linked_actions[-1].append(action_index)
                else:
                    embedded_actions[-1].append(global_action_index)
        num_embedded_actions = max(len(actions) for actions in embedded_actions)
        num_linked_actions = max(len(actions) for actions in linked_actions)
        if num_linked_actions == 0:
            linked_actions = None
        considered_actions: List[List[int]] = []
        for global_action_list in global_valid_actions:
            considered_actions.append([])
            # First we add the embedded actions to the list.
            for global_action_index, action_index in global_action_list:
                if global_action_index != -1:
                    considered_actions[-1].append(action_index)
            # Then we pad that portion.
            while len(considered_actions[-1]) < num_embedded_actions:
                considered_actions[-1].append(-1)
            # Then we add the linked actions to the list.
            for global_action_index, action_index in global_action_list:
                if global_action_index == -1:
                    considered_actions[-1].append(action_index)
            # Finally, we pad the linked portion.
            while len(considered_actions[-1]) < num_embedded_actions + num_linked_actions:
                considered_actions[-1].append(-1)
        return considered_actions, embedded_actions, linked_actions

    @staticmethod
    def _get_action_embeddings(state: WikiTablesDecoderState,
                               actions_to_embed: List[List[int]]) -> Tuple[torch.Tensor,
                                                                           torch.Tensor,
                                                                           torch.Tensor,
                                                                           torch.Tensor]:
        """
        Returns an embedded representation for all actions in ``actions_to_embed``, using the state
        in ``WikiTablesDecoderState``.

        Parameters
        ----------
        state : ``WikiTablesDecoderState``
            The current state.  We'll use this to get the global action embeddings.
        actions_to_embed : ``List[List[int]]``
            A list of _global_ action indices for each group element.  Should have shape
            (group_size, num_actions), unpadded.  This is expected to be output from
            :func:`_get_actions_to_consider`.

        Returns
        -------
        action_embeddings : ``torch.FloatTensor``
            An embedded representation of all of the given actions.  Shape is ``(group_size,
            num_actions, action_embedding_dim)``, where ``num_actions`` is the maximum number of
            considered actions for any group element.
        output_action_embeddings : ``torch.FloatTensor``
            A second embedded representation of all of the given actions.  The first is used when
            selecting actions, the second is used as the decoder output (which is the input at the
            next timestep).  This is similar to having separate word embeddings and softmax layer
            weights in a language model or MT model.
        action_biases : ``torch.FloatTensor``
            A bias weight for predicting each action.  Shape is ``(group_size, num_actions, 1)``.
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

        flattened_output_embeddings = state.output_action_embeddings.index_select(0, flattened_actions)
        output_embeddings = flattened_output_embeddings.view(group_size, max_num_actions, action_embedding_dim)

        flattened_biases = state.action_biases.index_select(0, flattened_actions)
        biases = flattened_biases.view(group_size, max_num_actions, 1)

        sequence_lengths = Variable(action_embeddings.data.new(num_actions))
        action_mask = util.get_mask_from_sequence_lengths(sequence_lengths, max_num_actions)
        return action_embeddings, output_embeddings, biases, action_mask

    def _get_entity_action_logits(self,
                                  state: WikiTablesDecoderState,
                                  actions_to_link: List[List[int]],
                                  attention_weights: torch.Tensor,
                                  linked_checklist_balance: torch.Tensor = None) -> Tuple[torch.FloatTensor,
                                                                                          torch.LongTensor,
                                                                                          torch.FloatTensor]:
        """
        Returns scores for each action in ``actions_to_link`` that are derived from the linking
        scores between the question and the table entities, and the current attention on the
        question.  The intuition is that if we're paying attention to a particular word in the
        question, we should tend to select entity productions that we think that word refers to.
        We additionally return a mask representing which elements in the returned ``action_logits``
        tensor are just padding, and an embedded representation of each action that can be used as
        input to the next step of the encoder.  That embedded representation is derived from the
        type of the entity produced by the action.

        The ``actions_to_link`` are in terms of the `batch` action list passed to
        ``model.forward()``.  We need to convert these integers into indices into the linking score
        tensor, which has shape (batch_size, num_entities, num_question_tokens), look up the
        linking score for each entity, then aggregate the scores using the current question
        attention.

        Parameters
        ----------
        state : ``WikiTablesDecoderState``
            The current state.  We'll use this to get the linking scores.
        actions_to_link : ``List[List[int]]``
            A list of _batch_ action indices for each group element.  Should have shape
            (group_size, num_actions), unpadded.  This is expected to be output from
            :func:`_get_actions_to_consider`.
        attention_weights : ``torch.Tensor``
            The current attention weights over the question tokens.  Should have shape
            ``(group_size, num_question_tokens)``.
        linked_checklist_balance : ``torch.Tensor``, optional (default=None)
            If the parser is being trained to maximize coverage over an agenda, this is the balance
            vector corresponding to entity actions, containing 1s and 0s, with 1s showing the
            actions that are yet to be produced. Required only if the parser is being trained to
            maximize coverage.

        Returns
        -------
        action_logits : ``torch.FloatTensor``
            A score for each of the given actions.  Shape is ``(group_size, num_actions)``, where
            ``num_actions`` is the maximum number of considered actions for any group element.
        action_mask : ``torch.LongTensor``
            A mask of shape ``(group_size, num_actions)`` indicating which ``(group_index,
            action_index)`` pairs were merely added as padding.
        type_embeddings : ``torch.LongTensor``
            A tensor of shape ``(group_size, num_actions, action_embedding_dim)``, with an embedded
            representation of the `type` of the entity corresponding to each action.
        """
        # First we map the actions to entity indices, using state.actions_to_entities, and find the
        # type of each entity using state.entity_types.
        action_entities: List[List[int]] = []
        entity_types: List[List[int]] = []
        for batch_index, action_list in zip(state.batch_indices, actions_to_link):
            action_entities.append([])
            entity_types.append([])
            for action_index in action_list:
                entity_index = state.actions_to_entities[(batch_index, action_index)]
                action_entities[-1].append(entity_index)
                entity_types[-1].append(state.entity_types[entity_index])

        # Then we create a padded tensor suitable for use with
        # `state.flattened_linking_scores.index_select()`.
        num_actions = [len(action_list) for action_list in action_entities]
        max_num_actions = max(num_actions)
        padded_actions = [common_util.pad_sequence_to_length(action_list, max_num_actions)
                          for action_list in action_entities]
        padded_types = [common_util.pad_sequence_to_length(type_list, max_num_actions)
                        for type_list in entity_types]
        # Shape: (group_size, num_actions)
        action_tensor = Variable(state.score[0].data.new(padded_actions).long())
        type_tensor = Variable(state.score[0].data.new(padded_types).long())

        # To get the type embedding tensor, we just use an embedding matrix on the list of entity
        # types.
        type_embeddings = self._entity_type_embedding(type_tensor)
        # `state.flattened_linking_scores` is shape (batch_size * num_entities, num_question_tokens).
        # We want to select from this using `action_tensor` to get a tensor of shape (group_size,
        # num_actions, num_question_tokens).  Unfortunately, the index_select functions in nn.util
        # don't do this operation.  So we'll do some reshapes and do the index_select ourselves.
        group_size = len(state.batch_indices)
        num_question_tokens = state.flattened_linking_scores.size(-1)
        flattened_actions = action_tensor.view(-1)
        # (group_size * num_actions, num_question_tokens)
        flattened_action_linking = state.flattened_linking_scores.index_select(0, flattened_actions)
        # (group_size, num_actions, num_question_tokens)
        action_linking = flattened_action_linking.view(group_size, max_num_actions, num_question_tokens)

        # Now we get action logits by weighting these entity x token scores by the attention over
        # the question tokens.  We can do this efficiently with torch.bmm.
        action_logits = action_linking.bmm(attention_weights.unsqueeze(-1)).squeeze(-1)
        if linked_checklist_balance is not None:
            # ``linked_checklist_balance`` is a binary tensor of size (group_size, num_actions) with
            # 1s indicating the linked actions that the agenda wants the decoder to produce, but
            # haven't been produced yet. We're simply doubling the logits of those actions here.
            action_logits_addition = action_logits * linked_checklist_balance
            action_logits = action_logits + self._linked_checklist_multiplier * action_logits_addition
        # Finally, we make a mask for our action logit tensor.
        sequence_lengths = Variable(action_linking.data.new(num_actions))
        action_mask = util.get_mask_from_sequence_lengths(sequence_lengths, max_num_actions)
        return action_logits, action_mask, type_embeddings

    @staticmethod
    def _compute_new_states(state: WikiTablesDecoderState,
                            log_probs: torch.Tensor,
                            hidden_state: torch.Tensor,
                            memory_cell: torch.Tensor,
                            action_embeddings: torch.Tensor,
                            attended_question: torch.Tensor,
                            attention_weights: torch.Tensor,
                            considered_actions: List[List[int]],
                            allowed_actions: List[Set[int]],
                            max_actions: int = None) -> List[WikiTablesDecoderState]:
        # Each group index here might get accessed multiple times, and doing the slicing operation
        # each time is more expensive than doing it once upfront.  These three lines give about a
        # 10% speedup in training time.  I also tried this with sorted_log_probs and
        # action_embeddings, but those get accessed for _each action_, so doing the splits there
        # didn't help.
        hidden_state = [x.squeeze(0) for x in hidden_state.split(1, 0)]
        memory_cell = [x.squeeze(0) for x in memory_cell.split(1, 0)]
        attended_question = [x.squeeze(0) for x in attended_question.split(1, 0)]

        sorted_log_probs, sorted_actions = log_probs.sort(dim=-1, descending=True)
        if max_actions is not None:
            # We might need a version of `sorted_log_probs` on the CPU later, but only if we need
            # to truncate the best states to `max_actions`.
            sorted_log_probs_cpu = sorted_log_probs.data.cpu().numpy()
        if state.debug_info is not None:
            probs_cpu = log_probs.exp().data.cpu().numpy().tolist()
        sorted_actions = sorted_actions.data.cpu().numpy().tolist()
        best_next_states: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        for group_index, (batch_index, group_actions) in enumerate(zip(state.batch_indices,
                                                                       sorted_actions)):
            for action_index, action in enumerate(group_actions):
                # `action` is currently the index in `log_probs`, not the actual action ID.  To get
                # the action ID, we need to go through `considered_actions`.
                action = considered_actions[group_index][action]
                if action == -1:
                    # This was padding.
                    continue
                if allowed_actions is not None and action not in allowed_actions[group_index]:
                    # This happens when our _decoder trainer_ wants us to only evaluate certain
                    # actions, likely because they are the gold actions in this state.  We just skip
                    # emitting any state that isn't allowed by the trainer, because constructing the
                    # new state can be expensive.
                    continue
                best_next_states[batch_index].append((group_index, action_index, action))
        new_states = []
        for batch_index, best_states in sorted(best_next_states.items()):
            if max_actions is not None:
                # We sorted previously by _group_index_, but we then combined by _batch_index_.  We
                # need to get the top next states for each _batch_ instance, so we sort all of the
                # instance's states again (across group index) by score.  We don't need to do this
                # if `max_actions` is None, because we'll be keeping all of the next states,
                # anyway.
                best_states.sort(key=lambda x: sorted_log_probs_cpu[x[:2]], reverse=True)
                best_states = best_states[:max_actions]
            for group_index, action_index, action in best_states:
                # We'll yield a bunch of states here that all have a `group_size` of 1, so that the
                # learning algorithm can decide how many of these it wants to keep, and it can just
                # regroup them later, as that's a really easy operation.
                batch_index = state.batch_indices[group_index]
                new_action_history = state.action_history[group_index] + [action]
                new_score = sorted_log_probs[group_index, action_index]

                # `action_index` is the index in the _sorted_ tensors, but the action embedding
                # matrix is _not_ sorted, so we need to get back the original, non-sorted action
                # index before we get the action embedding.
                action_embedding_index = sorted_actions[group_index][action_index]
                action_embedding = action_embeddings[group_index, action_embedding_index, :]
                production_rule = state.possible_actions[batch_index][action][0]
                new_grammar_state = state.grammar_state[group_index].take_action(production_rule)
                if state.checklist_state[0] is not None:
                    new_checklist_state = [state.checklist_state[group_index].update(action)]
                else:
                    new_checklist_state = None
                if state.debug_info is not None:
                    debug_info = {
                            'considered_actions': considered_actions[group_index],
                            'question_attention': attention_weights[group_index],
                            'probabilities': probs_cpu[group_index],
                            }
                    new_debug_info = [state.debug_info[group_index] + [debug_info]]
                else:
                    new_debug_info = None

                new_rnn_state = RnnState(hidden_state[group_index],
                                         memory_cell[group_index],
                                         action_embedding,
                                         attended_question[group_index],
                                         state.rnn_state[group_index].encoder_outputs,
                                         state.rnn_state[group_index].encoder_output_mask)
                new_state = WikiTablesDecoderState(batch_indices=[batch_index],
                                                   action_history=[new_action_history],
                                                   score=[new_score],
                                                   rnn_state=[new_rnn_state],
                                                   grammar_state=[new_grammar_state],
                                                   action_embeddings=state.action_embeddings,
                                                   output_action_embeddings=state.output_action_embeddings,
                                                   action_biases=state.action_biases,
                                                   action_indices=state.action_indices,
                                                   possible_actions=state.possible_actions,
                                                   flattened_linking_scores=state.flattened_linking_scores,
                                                   actions_to_entities=state.actions_to_entities,
                                                   entity_types=state.entity_types,
                                                   world=state.world,
                                                   example_lisp_string=state.example_lisp_string,
                                                   checklist_state=new_checklist_state,
                                                   debug_info=new_debug_info)
                new_states.append(new_state)
        return new_states
