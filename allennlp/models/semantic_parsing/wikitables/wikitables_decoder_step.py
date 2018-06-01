from collections import defaultdict
from typing import Dict, List, Set, Tuple

from overrides import overrides

import torch
from torch.nn import Parameter
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear

from allennlp.common import util as common_util
from allennlp.common.checks import check_dimensions_match
from allennlp.models.semantic_parsing.wikitables.wikitables_decoder_state import WikiTablesDecoderState
from allennlp.modules import Attention, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.decoding import DecoderStep, RnnState


class WikiTablesDecoderStep(DecoderStep[WikiTablesDecoderState]):
    """
    Parameters
    ----------
    encoder_output_dim : ``int``
    action_embedding_dim : ``int``
    input_attention : ``Attention``
    num_start_types : ``int``
    predict_start_type_separately : ``bool``, optional (default=True)
        If ``True``, we will predict the initial action (which is typically the base type of the
        logical form) using a different mechanism than our typical action decoder.  We basically
        just do a projection of the hidden state, and don't update the decoder RNN.
    add_action_bias : ``bool``, optional (default=True)
        If ``True``, there has been a bias dimension added to the embedding of each action, which
        gets used when predicting the next action.  We add a dimension of ones to our predicted
        action vector in this case to account for that.
    mixture_feedforward : ``FeedForward`` optional (default=None)
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
                 input_attention: Attention,
                 num_start_types: int,
                 predict_start_type_separately: bool = True,
                 add_action_bias: bool = True,
                 mixture_feedforward: FeedForward = None,
                 dropout: float = 0.0,
                 unlinked_terminal_indices: List[int] = None) -> None:
        super(WikiTablesDecoderStep, self).__init__()
        self._mixture_feedforward = mixture_feedforward
        self._input_attention = input_attention
        self._add_action_bias = add_action_bias

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
        if self._predict_start_type_separately and not state.action_history[0]:
            # The wikitables parser did something different when predicting the start type, which
            # is our first action.  So in this case we break out into a different function.  We'll
            # ignore max_actions on our first step, assuming there aren't that many start types.
            return self._take_first_step(state, allowed_actions)

        # This method is long and involved, but because of some closures we construct, it's better
        # to keep it one method than to separate it out into several.  We'll at least group it into
        # sections; here's a brief outline of the sections: We'll (1) construct the input to the
        # decoder and update the decoder's hidden state.  Then we'll (2) use this new hidden state
        # (and maybe other information) to predict an action.  Finally, we will (3) construct new
        # states for the next step.  Each new state corresponds to one valid action that can be
        # taken from the current state, and they are ordered by their probability of being
        # selected.

        #########################
        # 1: Updating the decoder
        #########################

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

        # (group_size, action_embedding_dim)
        projected_query = torch.nn.functional.relu(self._output_projection_layer(action_query))
        predicted_action_embeddings = self._dropout(projected_query)
        if self._add_action_bias:
            # NOTE: It's important that this happens right before the dot product with the action
            # embeddings.  Otherwise this isn't a proper bias.  We do it here instead of right next
            # to the `.mm` below just so we only do it once for the whole group.
            ones = predicted_action_embeddings.new([[1] for _ in range(group_size)])
            predicted_action_embeddings = torch.cat([predicted_action_embeddings, ones], dim=-1)

        ################################
        # 2: Predicting the next actions
        ################################

        # In this section we take our predicted action embedding and compare it to the available
        # actions in our current state (which might be different for each group element).  For
        # computing action scores, we'll forget about doing batched / grouped computation, as it
        # adds too much complexity and doesn't speed things up, anyway, with the operations we're
        # doing here.  This means we don't need any action masks, as we'll only get the right
        # lengths for what we're computing.

        # TODO(mattg): Maybe this section could reasonably be pulled out into its own method...

        actions = state.get_valid_actions()

        batch_results = defaultdict(list)
        for group_index in range(group_size):
            instance_actions = actions[group_index]
            predicted_action_embedding = predicted_action_embeddings[group_index]
            action_embeddings, output_action_embeddings, embedded_actions = instance_actions['global']
            # TODO(mattg): Add back in the checklist balance modification to the predicted action
            # embedding.
            # This is just a matrix product between a (num_actions, embedding_dim) matrix and an
            # (embedding_dim, 1) matrix.
            embedded_action_logits = action_embeddings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
            instance_action_ids = embedded_actions[:]
            if 'linked' in instance_actions:
                linking_scores, type_embeddings, linked_actions = instance_actions['linked']
                instance_action_ids.extend(linked_actions)
                # TODO(mattg): Add back in the checklist balance addition to the linked logits.
                # Another matrix product, this time (num_actions, num_question_tokens) x
                # (num_question_tokens, 1)
                linked_action_logits = linking_scores.mm(attention_weights[group_index].unsqueeze(-1)).squeeze(-1)

                # The `output_action_embeddings` tensor gets used later as the input to the next
                # decoder step.  For linked actions, we don't have any action embedding, so we use
                # the entity type instead.
                output_action_embeddings = torch.cat([output_action_embeddings, type_embeddings], dim=0)

                if self._mixture_feedforward is not None:
                    # The linked and global logits are combined with a mixture weight to prevent the
                    # linked_action_logits from dominating the embedded_action_logits if a softmax
                    # was applied on both together.
                    mixture_weight = self._mixture_feedforward(hidden_state)
                    mix1 = torch.log(mixture_weight)
                    mix2 = torch.log(1 - mixture_weight)

                    entity_action_probs = torch.nn.functional.log_softmax(linked_action_logits, dim=-1) + mix1
                    embedded_action_probs = torch.nn.functional.log_softmax(embedded_action_logits, dim=-1) + mix2
                    current_log_probs = torch.cat([embedded_action_probs, entity_action_probs], dim=-1)
                else:
                    action_logits = torch.cat([embedded_action_logits, linked_action_logits], dim=-1)
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
                                                                    output_action_embeddings,
                                                                    instance_action_ids))

        ############################
        # 3: Constructing new states
        ############################

        # We'll yield a bunch of states here that all have a `group_size` of 1, so that the
        # learning algorithm can decide how many of these it wants to keep, and it can just regroup
        # them later, as that's a really easy operation.
        #
        # We first define a `make_state` method, as in the logic that follows we want to create
        # states in a couple of different branches, and we don't want to duplicate the
        # state-creation logic.  This method creates a closure using the variables computed above,
        # so it doesn't make sense to pull it out of here.

        # Each group index here might get accessed multiple times, and doing the slicing operation
        # each time is more expensive than doing it once upfront.  These three lines give about a
        # 10% speedup in training time.
        hidden_state = [x.squeeze(0) for x in hidden_state.split(1, 0)]
        memory_cell = [x.squeeze(0) for x in memory_cell.split(1, 0)]
        attended_question = [x.squeeze(0) for x in attended_question.split(1, 0)]

        def make_state(group_index: int,
                       action: int,
                       new_score: torch.Tensor,
                       action_embedding: torch.Tensor) -> WikiTablesDecoderState:
            batch_index = state.batch_indices[group_index]
            new_action_history = state.action_history[group_index] + [action]
            production_rule = state.possible_actions[batch_index][action][0]
            new_grammar_state = state.grammar_state[group_index].take_action(production_rule)
            if state.checklist_state[0] is not None:
                new_checklist_state = [state.checklist_state[group_index].update(action)]
            else:
                new_checklist_state = None
            if state.debug_info is not None:
                considered_actions = []
                for i, log_probs, _, actions in batch_results[batch_index]:
                    if i == group_index:
                        considered_actions = actions
                        probabilities = log_probs.exp().cpu()
                debug_info = {
                        'considered_actions': considered_actions,
                        'question_attention': attention_weights[group_index],
                        'probabilities': probabilities,
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
                                               possible_actions=state.possible_actions,
                                               world=state.world,
                                               example_lisp_string=state.example_lisp_string,
                                               checklist_state=new_checklist_state,
                                               debug_info=new_debug_info)
            return new_state

        new_states = []
        for batch_index, results in batch_results.items():
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
                group_log_probs = []
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
        terminal_indices_tensor = state.score[0].new_tensor(group_terminal_indices, dtype=torch.long)
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
        log_probs = torch.nn.functional.log_softmax(start_action_logits, dim=-1)
        sorted_log_probs, sorted_actions = log_probs.sort(dim=-1, descending=True)

        sorted_actions = sorted_actions.detach().cpu().numpy().tolist()
        if state.debug_info is not None:
            probs_cpu = log_probs.exp().detach().cpu().numpy().tolist()

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
                                                   possible_actions=state.possible_actions,
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
