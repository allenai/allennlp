
class NlvrDecoderState(DecoderState['NlvrDecoderState']):
    """
    This class is very similar to ``WikiTablesDecoderState``, except that we keep track of a
    checklist score, and other variables related to it.

    Parameters
    ----------
    terminal_actions : ``List[torch.Tensor]``
        Each element in the list is a vector containing the indices of terminal actions. Currently
        the vectors are the same for all instances, because we consider all terminals for each
        instance. In the future, we may want to include only world-specific terminal actions here.
        Each of these vectors is needed for computing checklists for next states.
    checklist_target : ``List[torch.LongTensor]``
        List of targets corresponding to agendas that indicate the states we want the checklists to
        ideally be. Each element in this list is the same size as the corresponding element in
        ``agenda_relevant_actions``, and it contains 1 for each corresponding action in the relevant
        actions list that we want to see in the final logical form, and 0 for each corresponding
        action that we do not.
    checklist_masks : ``List[torch.Tensor]``
        Masks corresponding to ``terminal_actions``, indicating which of those actions are relevant
        for checklist computation. For example, if the parser is penalizing non-agenda terminal
        actions, all the terminal actions are relevant.
    checklist : ``List[Variable]``
        A checklist for each instance indicating how many times each action in its agenda has
        been chosen previously. It contains the actual counts of the agenda actions.
    checklist_cost_weight : ``float``
        The cost associated with each state has two components, one based on how well its action
        sequence covers the agenda, and the other based on whether the sequence evaluates to the
        correct denotation. The final cost is a linear combination of the two, and this weight is
        the one associated with the checklist cost.
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
    attended_sentence : ``List[torch.Tensor]``
        This holds the attention-weighted sum over the sentence representations that we computed in
        the previous timestep, for each element in the group.  We keep this as part of the state
        because we use the previous attention as part of our decoder cell update.  Each tensor in
        this list has shape ``(encoder_output_dim,)``.
    grammar_state : ``List[GrammarState]``
        This hold the current grammar state for each element of the group.  The ``GrammarState``
        keeps track of which actions are currently valid.
    encoder_outputs : ``List[torch.Tensor]``
        A list of variables, each of shape ``(sentence_length, encoder_output_dim)``, containing
        the encoder outputs at each timestep.  The list is over batch elements, and we do the input
        this way so we can easily do a ``torch.cat`` on a list of indices into this batched list.

        Note that all of the above lists are of length ``group_size``, while the encoder outputs
        and mask are lists of length ``batch_size``.  We always pass around the encoder outputs and
        mask unmodified, regardless of what's in the grouping for this state.  We'll use the
        ``batch_indices`` for the group to pull pieces out of these lists when we're ready to
        actually do some computation.
    encoder_output_mask : ``List[torch.Tensor]``
        A list of variables, each of shape ``(sentence_length,)``, containing a mask over sentence
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
    world : ``List[NlvrWorld]``
        The world associated with each element. This is needed to compute the denotations.
    label_strings : ``List[str]``
        String representations of labels for the elements provided. When scoring finished states, we
        will compare the denotations of their action sequences against these labels.
    """
    def __init__(self,
                 terminal_actions: List[torch.Tensor],
                 checklist_target: List[torch.Tensor],
                 checklist_masks: List[torch.Tensor],
                 checklist: List[Variable],
                 checklist_cost_weight: float,
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor],
                 hidden_state: List[torch.Tensor],
                 memory_cell: List[torch.Tensor],
                 previous_action_embedding: List[torch.Tensor],
                 attended_sentence: List[torch.Tensor],
                 grammar_state: List[GrammarState],
                 encoder_outputs: List[torch.Tensor],
                 encoder_output_mask: List[torch.Tensor],
                 action_embeddings: torch.Tensor,
                 action_indices: Dict[Tuple[int, int], int],
                 possible_actions: List[List[ProductionRuleArray]],
                 worlds: List[NlvrWorld],
                 label_strings: List[str]) -> None:
        super(NlvrDecoderState, self).__init__(batch_indices, action_history, score)
        self.terminal_actions = terminal_actions
        self.checklist_target = checklist_target
        self.checklist_mask = checklist_masks
        self.checklist = checklist
        self.checklist_cost_weight = checklist_cost_weight
        self.hidden_state = hidden_state
        self.memory_cell = memory_cell
        self.previous_action_embedding = previous_action_embedding
        self.attended_sentence = attended_sentence
        self.grammar_state = grammar_state
        self.encoder_outputs = encoder_outputs
        self.encoder_output_mask = encoder_output_mask
        self.action_embeddings = action_embeddings
        self.action_indices = action_indices
        self.possible_actions = possible_actions
        self.worlds = worlds
        self.label_strings = label_strings

    def get_valid_actions(self) -> List[List[int]]:
        """
        Returns a list of valid actions for each element of the group.
        """
        valid_actions = [state.get_valid_actions() for state in self.grammar_state]
        return valid_actions

    def denotation_is_correct(self) -> bool:
        """
        Returns whether action history in the state evaluates to the correct denotation. Only
        defined when the state is finished.
        """
        assert self.is_finished(), "Cannot compute denotations for unfinished states!"
        # Since this is a finished state, its group size must be 1.
        batch_index = self.batch_indices[0]
        world = self.worlds[batch_index]
        label_string = self.label_strings[batch_index]
        history = self.action_history[0]
        action_sequence = [self._get_action_string(action) for action in history]
        logical_form = world.get_logical_form(action_sequence)
        denotation = world.execute(logical_form)
        is_correct = str(denotation).lower() == label_string.lower()
        return is_correct

    def get_state_info(self) -> Dict[str, List]:
        """
        This method is here for debugging purposes, in case you want to look at the what the model
        is learning. It may be inefficient to call it while training the model on real data.
        """
        if len(self.batch_indices) == 1 and self.is_finished():
            costs = [float(self.get_cost().data.cpu().numpy())]
        else:
            costs = []
        model_scores = [float(score.data.cpu().numpy()) for score in self.score]
        action_sequences = [[self._get_action_string(action) for action in history]
                            for history in self.action_history]
        agenda_sequences = []
        all_agenda_indices = []
        for agenda, checklist_target in zip(self.terminal_actions, self.checklist_target):
            agenda_indices = []
            for action, is_wanted in zip(agenda, checklist_target):
                action_int = int(action.data.cpu().numpy())
                is_wanted_int = int(is_wanted.data.cpu().numpy())
                if is_wanted_int != 0:
                    agenda_indices.append(action_int)
            agenda_sequences.append([self._get_action_string(action) for action in agenda_indices])
            all_agenda_indices.append(agenda_indices)
        return {"agenda": agenda_sequences,
                "agenda_indices": all_agenda_indices,
                "history": action_sequences,
                "history_indices": self.action_history,
                "costs": costs,
                "scores": model_scores}

    def _get_action_string(self, action_id: int) -> str:
        # Possible actions for all worlds are the same.
        all_actions = self.possible_actions[0]
        return "%s -> %s" % (all_actions[action_id]["left"][0],
                             all_actions[action_id]["right"][0])

    def get_checklist_balances(self) -> List[Variable]:
        """
        Returns a list of checklist balances for this state. Each balance is a float vector
        containing just 1s and 0s showing which of the items are filled. We clamp the min at 0 to
        ignore the number of times an action is taken. The value at an index will be 1 iff the
        target wants an unmasked action to be taken, and it is not yet taken. All elements in each
        balance corresponding to masked actions will be 0.
        """
        checklist_balances: List[Variable] = []
        for instance_target, instance_checklist, checklist_mask in zip(self.checklist_target,
                                                                       self.checklist,
                                                                       self.checklist_mask):
            checklist_balance = torch.clamp(instance_target - instance_checklist, min=0.0)
            checklist_balance = checklist_balance * checklist_mask
            checklist_balances.append(checklist_balance)
        return checklist_balances

    def get_cost(self) -> Variable:
        """
        Return the costs a finished state. Since it is a finished state, the group size will be 1,
        and hence we'll return just one cost.
        """
        if not self.is_finished():
            raise RuntimeError("get_costs() is not defined for unfinished states!")
        instance_checklist_target = self.checklist_target[0]
        instance_checklist = self.checklist[0]
        instance_checklist_mask = self.checklist_mask[0]
        checklist_cost = - self.score_single_checklist(instance_checklist,
                                                       instance_checklist_target,
                                                       instance_checklist_mask)
        # This is the number of items on the agenda that we want to see in the decoded sequence.
        # We use this as the denotation cost if the path is incorrect.
        # Note: If we are penalizing the model for producing non-agenda actions, this is not the
        # upper limit on the checklist cost. That would be the number of terminal actions.
        denotation_cost = torch.sum(instance_checklist_target.float())
        checklist_cost = self.checklist_cost_weight * checklist_cost
        if self.denotation_is_correct():
            cost = checklist_cost
        else:
            cost = checklist_cost + (1 - self.checklist_cost_weight) * denotation_cost
        return cost

    @classmethod
    def score_single_checklist(cls,
                               instance_checklist: Variable,
                               instance_checklist_target: Variable,
                               instance_checklist_mask: Variable) -> Variable:
        """
        Takes a single checklist, a corresponding checklist target and a mask, and returns
        the score of the checklist. We want the checklist to be as close to the target as possible
        for the unmasked elements.
        """
        balance = instance_checklist_target - instance_checklist
        balance = balance * instance_checklist_mask
        return -torch.sum((balance) ** 2)

    def is_finished(self) -> bool:
        """This method is identical to ``WikiTablesDecoderState.is_finished``."""
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_state[0].is_finished()

    @classmethod
    def combine_states(cls, states) -> 'NlvrDecoderState':
        terminal_actions = [actions for state in states for actions in state.terminal_actions]
        checklist_target = [target_list for state in states for target_list in
                            state.checklist_target]
        checklist_masks = [mask for state in states for mask in state.checklist_mask]
        checklist = [checklist_list for state in states for checklist_list in state.checklist]
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        hidden_states = [hidden_state for state in states for hidden_state in state.hidden_state]
        memory_cells = [memory_cell for state in states for memory_cell in state.memory_cell]
        previous_action = [action for state in states for action in state.previous_action_embedding]
        attended_sentence = [attended for state in states for attended in state.attended_sentence]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        return NlvrDecoderState(terminal_actions,
                                checklist_target,
                                checklist_masks,
                                checklist,
                                states[0].checklist_cost_weight,
                                batch_indices,
                                action_histories,
                                scores,
                                hidden_states,
                                memory_cells,
                                previous_action,
                                attended_sentence,
                                grammar_states,
                                states[0].encoder_outputs,
                                states[0].encoder_output_mask,
                                states[0].action_embeddings,
                                states[0].action_indices,
                                states[0].possible_actions,
                                states[0].worlds,
                                states[0].label_strings)
