from typing import List, Set, Dict, Tuple
from collections import defaultdict

from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear

from allennlp.common import Params
from allennlp.common import util as common_util
from allennlp.common.checks import check_dimensions_match
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.data.semparse.type_declarations.type_declaration import START_SYMBOL
from allennlp.data.semparse.type_declarations import GrammarState
from allennlp.data.semparse.worlds import NlvrWorld
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.decoding import DecoderState, DecoderStep, DecoderTrainer
from allennlp.nn import util as nn_util
from allennlp.models.model import Model
from allennlp.training.metrics import Average


@Model.register("nlvr_parser")
class NlvrSemanticParser(Model):
    """
    ``NlvrSemanticParser`` is a semantic parsing model built for the NLVR domain.  There is a lot of
    overlap with ``WikiTablesSemanticParser`` here. We may want to eventually move the common
    functionality into a more general transition-based parsing class.

    The main differences between this parser and what we have for Wikitables are that we have an
    agenda of actions instead of complete target action sequences, and accordingly the score in this
    parser is based on how many of the agenda actions are covered.

    This is still WORK IN PROGRESS. We still need to incorporate other kinds of losses, including
    atleast a denotation based loss.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 nonterminal_embedder: TextFieldEmbedder,
                 terminal_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder_trainer: DecoderTrainer,
                 max_decoding_steps: int,
                 attention_function: SimilarityFunction) -> None:
        super(NlvrSemanticParser, self).__init__(vocab=vocab)

        self._sentence_embedder = sentence_embedder
        self._action_sequence_validity = Average()
        self._denotation_accuracy = Average()
        check_dimensions_match(nonterminal_embedder.get_output_dim(),
                               terminal_embedder.get_output_dim(),
                               "nonterminal embedding dim",
                               "terminal embedding dim")
        self._nonterminal_embedder = nonterminal_embedder
        self._terminal_embedder = terminal_embedder
        self._encoder = encoder
        self._decoder_trainer = decoder_trainer
        self._max_decoding_steps = max_decoding_steps
        self._attention_function = attention_function
        action_embedding_dim = nonterminal_embedder.get_output_dim() * 2

        # self._decoder_step would be set to ``WikiTablesDecoderStep``. Rewriting it.
        self._decoder_step = NlvrDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                             action_embedding_dim=action_embedding_dim,
                                             attention_function=attention_function)
    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                world: List[NlvrWorld],
                actions: List[List[ProductionRuleArray]],
                agenda: torch.LongTensor,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ,unused-argument
        """
        Decoder logic for producing type constrained target sequences, that maximize coverage of
        their respective agendas. This will change soon, to include a denotation based score as
        well, once we have a way to transform action sequences into logical forms that can be
        executed to produce denotations.
        """
        # TODO(pradeep): Use labels.
        embedded_input = self._sentence_embedder(sentence)
        # (batch_size, sentence_length)
        sentence_mask = nn_util.get_text_field_mask(sentence).float()

        batch_size = embedded_input.size(0)

        # (batch_size, sentence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, sentence_mask)

        # These are the indices of the last words in the sentences (i.e. length sans padding - 1).
        # We are assuming sentences are right padded.
        # (batch_size,)
        last_word_indices = sentence_mask.sum(1).long() - 1
        batch_size, _, encoder_output_dim = encoder_outputs.size()
        # Expanding indices to 3 dimensions
        expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
        # (batch_size, 1, encoder_output_dim)
        final_encoder_output = encoder_outputs.gather(1, expanded_indices)
        final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
        memory_cell = nn_util.new_variable_with_size(encoder_outputs,
                                                     (batch_size, self._encoder.get_output_dim()),
                                                     0)
        attended_sentence = self._decoder_step.attend_on_sentence(final_encoder_output,
                                                                  encoder_outputs, sentence_mask)
        action_embeddings, action_indices, initial_action_embedding = self._embed_actions(actions)
        # Get a mapping from production rules to global action ids.
        agenda_mask = agenda != -1
        agenda_mask = agenda_mask.long()
        # (batch_size, agenda_size)
        initial_checklist = nn_util.new_variable_with_size(agenda, agenda.size(), 0).float()
        agenda_list = [agenda[i] for i in range(batch_size)]
        agenda_mask_list = [agenda_mask[i] for i in range(batch_size)]
        initial_checklist_list = [initial_checklist[i] for i in range(batch_size)]
        initial_score_list = [NlvrDecoderStep.score_instance_checklist(checklist, agenda_mask)
                              for checklist, agenda_mask in zip(initial_checklist_list,
                                                                agenda_mask_list)]
        initial_hidden_state = [final_encoder_output[i] for i in range(batch_size)]
        initial_memory_cell = [memory_cell[i] for i in range(batch_size)]
        initial_action_embedding_list = [initial_action_embedding for _ in range(batch_size)]
        initial_grammar_state = [self._create_grammar_state(world[i], actions[i]) for i in
                                 range(batch_size)]
        initial_attended_sentence = [attended_sentence[i] for i in range(batch_size)]
        encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
        sentence_mask_list = [sentence_mask[i] for i in range(batch_size)]
        initial_state = NlvrDecoderState(agenda_list,
                                         agenda_mask_list,
                                         initial_checklist_list,
                                         list(range(batch_size)),
                                         [[] for _ in range(batch_size)],
                                         initial_score_list,
                                         initial_hidden_state,
                                         initial_memory_cell,
                                         initial_action_embedding_list,
                                         initial_attended_sentence,
                                         initial_grammar_state,
                                         encoder_outputs_list,
                                         sentence_mask_list,
                                         action_embeddings,
                                         action_indices,
                                         actions)

        outputs = self._decoder_trainer.decode(initial_state, self._decoder_step,  # type: ignore
                                               self._max_decoding_steps)

        return outputs

    @staticmethod
    def _create_grammar_state(world: NlvrWorld,
                              possible_actions: List[ProductionRuleArray]) -> GrammarState:
        valid_actions = world.get_valid_actions()
        action_mapping = {}
        for i, action in enumerate(possible_actions):
            action_string = action['left'][0] + ' -> ' + action['right'][0]
            action_mapping[action_string] = i
        translated_valid_actions = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = [action_mapping[action_string]
                                             for action_string in action_strings]
        return GrammarState([START_SYMBOL], {}, translated_valid_actions, action_mapping)

    def _embed_actions(self, actions: List[List[ProductionRuleArray]]) -> Tuple[torch.Tensor,
                                                                                Dict[Tuple[int, int], int],
                                                                                torch.Tensor]:
        """
        This method is almost identical to ``WikiTablesParser._embed_actions``. The only difference
        is we always embed terminals here.
        Given all of the possible actions for all batch instances, produce an embedding for them.
        There will be significant overlap in this list, as the production rules from the grammar
        are shared across all batch instances.  Our returned tensor has an embedding for each
        `unique` action, so we also need to return a mapping from the original ``(batch_index,
        action_index)`` to our new ``global_action_index``, so that we can get the right action
        embedding during decoding.

        Returns
        -------
        action_embeddings : ``torch.Tensor``
            Has shape ``(num_unique_actions, action_embedding_dim)``.
        action_map : ``Dict[Tuple[int, int], int]``
            Maps ``(batch_index, action_index)`` in the input action list to ``action_index`` in
            the ``action_embeddings`` tensor.
        initial_action_embedding : ``torch.Tensor``
            Has shape ``(action_embedding_dim,)``.  An embedding for the initial action input.
            This needs to be computed here, because we don't keep around the nonterminal
            embeddings.  We do this by creating a fake rule "0 -> START", where the LHS embedding
            is a vector of zeros, and the RHS embedding is the START symbol embedding.
        """
        # Basic outline: we'll embed actions by embedding their left hand side (LHS) and right hand
        # side (RHS) separately, then concatenating the two parts.  So first we need to find all of
        # the unique terminals and non-terminals in the production rules, and embed those (for ease
        # of reference, we'll refer to non-terminals and terminals collectively as "elements" in
        # the logic below).  Then we'll gather all unique _actions_, and for each action, we'll use
        # an `index_select` to look up the embedding for it's LHS and RHS, then do the concat.
        nonterminals, terminals = self._get_unique_elements(actions)
        nonterminal_strings = sorted(nonterminals.keys())
        terminal_strings = sorted(terminals.keys())
        nonterminal_tensor_dicts = [nonterminals[key] for key in nonterminal_strings]
        terminal_tensor_dicts = [terminals[key] for key in terminal_strings]
        nonterminal_tensors = nn_util.batch_tensor_dicts(nonterminal_tensor_dicts,
                                                         remove_trailing_dimension=True)
        terminal_tensors = nn_util.batch_tensor_dicts(terminal_tensor_dicts,
                                                      remove_trailing_dimension=True)

        # The TextFieldEmbedder expects a 3D tensor, but we have 2D tensors, so we unsqueeze here
        # and squeeze after the embedding.
        nonterminal_tensors = {key: tensor.unsqueeze(0) for key, tensor in nonterminal_tensors.items()}
        terminal_tensors = {key: tensor.unsqueeze(0) for key, tensor in terminal_tensors.items()}
        # Shape: (num_nonterminals, element_embedding_dim)
        embedded_nonterminals = self._nonterminal_embedder(nonterminal_tensors).squeeze(0)

        # Shape: (num_terminals, element_embedding_dim)
        embedded_terminals = self._terminal_embedder(terminal_tensors).squeeze(0)
        # Shape: (num_nonterminals + num_terminals, element_embedding_dim)
        embedded_elements = torch.cat([embedded_nonterminals, embedded_terminals], dim=0)
        # This will map element strings to their index in the `embedded_elements` tensor.
        element_ids = {nonterminal: i for i, nonterminal in enumerate(nonterminal_strings)}
        element_ids.update({terminal: i + len(nonterminal_strings)
                            for i, terminal in enumerate(terminal_strings)})
        unique_nonterminal_actions: Set[Tuple[int, int]] = set()
        unique_terminal_actions: Set[Tuple[int, int]] = set()
        for instance_actions in actions:
            for action in instance_actions:
                if not action['left'][0]:
                    # This rule is padding.
                    continue
                # This gives us the LHS and RHS strings, which we map to ids in the element tensor.
                action_ids = (element_ids[action['left'][0]], element_ids[action['right'][0]])
                if action['right'][1]:
                    unique_nonterminal_actions.add(action_ids)
                else:
                    unique_terminal_actions.add(action_ids)
        unique_action_list = list(unique_nonterminal_actions) + list(unique_terminal_actions)
        action_left_sides, action_right_sides = zip(*unique_action_list)
        # We need a tensor to copy so we can create stuff on the right device; just grabbing one
        # from the nonterminals here.
        copy_tensor = list(list(nonterminals.values())[0].values())[0]
        action_left_indices = Variable(copy_tensor.data.new(list(action_left_sides)).long())
        action_right_indices = Variable(copy_tensor.data.new(list(action_right_sides)).long())
        left_side_embeddings = embedded_elements.index_select(0, action_left_indices)
        right_side_embeddings = embedded_elements.index_select(0, action_right_indices)
        # Shape: (num_actions, element_embedding_dim * 2)
        embedded_actions = torch.cat([left_side_embeddings, right_side_embeddings], dim=-1)

        # Next we'll construct the embedding for the initial action, which is a concatenation of a
        # zero LHS vector and the START RHS vector.
        zeros = Variable(copy_tensor.data.new(embedded_elements.size(-1)).fill_(0).float())
        start_vector = embedded_elements[element_ids[START_SYMBOL]]
        # Shape: (element_embedding_dim * 2,)
        initial_action_embedding = torch.cat([zeros, start_vector], dim=-1)

        # Now we just need to make a map from `(batch_index, action_index)` to `action_index`.
        # global_action_ids has the list of all unique actions; here we're going over all of the
        # actions for each batch instance so we can map them to the global action ids.
        global_action_ids = {action: i for i, action in enumerate(unique_action_list)}
        action_map: Dict[Tuple[int, int], int] = {}
        for batch_index, action_list in enumerate(actions):
            for action_index, action in enumerate(action_list):
                if not action['left'][0]:
                    # This rule is padding.
                    continue
                action_indices = (element_ids[action['left'][0]], element_ids[action['right'][0]])
                action_id = global_action_ids[action_indices]
                action_map[(batch_index, action_index)] = action_id
        return embedded_actions, action_map, initial_action_embedding

    @staticmethod
    def _get_unique_elements(
            actions: List[List[ProductionRuleArray]]) -> Tuple[Dict[str, Dict[str, torch.Tensor]],
                                                               Dict[str, Dict[str, torch.Tensor]]]:
        """
        This method is identical to ``WikiTablesSemanticParser._get_unique_elements``
        Finds all of the unique terminals and non-terminals in all of the production rules.  We
        will embed these elements separately, then use those embeddings to get final action
        embeddings.

        Returns
        -------
        nonterminals : ``Dict[str, Dict[str, torch.Tensor]]]``
            Each item in this dictionary represents a single nonterminal element of the grammar,
            like "d", "<r,d>", "or "<#1,#1>".  The key is the string representation of the
            nonterminal and the value its indexed representation, which we will use for computing
            the embedding.
        terminals : ``Dict[str, Dict[str, torch.Tensor]]]``
            Identical to ``nonterminals``, but for terminal elements of the grammar, like
            "fb:type.object.type", "reverse", or "fb:cell.2nd".
        """
        nonterminals: Dict[str, Dict[str, torch.Tensor]] = {}
        terminals: Dict[str, Dict[str, torch.Tensor]] = {}
        for action_sequence in actions:
            for production_rule in action_sequence:
                if not production_rule['left'][0]:
                    # This rule is padding.
                    continue
                # This logic is hard to understand, because the ProductionRuleArray is a messy
                # type.  The structure of each ProductionRuleArray is:
                #     {
                #      "left": (LHS_string, left_is_nonterminal, padded_LHS_tensor_dict),
                #      "right": (RHS_string, right_is_nonterminal, padded_RHS_tensor_dict)
                #     }
                # Technically, the left hand side is _always_ a non-terminal (by definition, you
                # can't expand a terminal), but we'll do this check anyway, in case you did
                # something really crazy.
                if production_rule['left'][1]:  # this is a nonterminal production
                    nonterminals[production_rule['left'][0]] = production_rule['left'][2]
                else:
                    terminals[production_rule['left'][0]] = production_rule['left'][2]
                if production_rule['right'][1]:  # this is a nonterminal production
                    nonterminals[production_rule['right'][0]] = production_rule['right'][2]
                else:
                    terminals[production_rule['right'][0]] = production_rule['right'][2]
        return nonterminals, terminals

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method is identical to ``WikiTablesSemanticParser.decode``
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions.  This is (confusingly) a separate notion from the "decoder"
        in "encoder/decoder", where that decoder logic lives in ``WikiTablesDecoderStep``.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        best_action_indices = output_dict["best_action_sequence"][0]
        action_strings = []
        for action_index in best_action_indices:
            action_strings.append(self.vocab.get_token_from_index(action_index,
                                                                  namespace=self._action_namespace))
        output_dict["predicted_actions"] = [action_strings]
        return output_dict

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'NlvrSemanticParser':
        sentence_embedder_params = params.pop("sentence_embedder")
        sentence_embedder = TextFieldEmbedder.from_params(vocab, sentence_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        nonterminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("nonterminal_embedder"))
        terminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("terminal_embedder"))
        decoder_trainer = DecoderTrainer.from_params(params.pop("decoder_trainer"))
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   sentence_embedder=sentence_embedder,
                   nonterminal_embedder=nonterminal_embedder,
                   terminal_embedder=terminal_embedder,
                   encoder=encoder,
                   decoder_trainer=decoder_trainer,
                   max_decoding_steps=max_decoding_steps,
                   attention_function=attention_function)


class NlvrDecoderState(DecoderState['NlvrDecoderState']):
    """
    This class is very similar to ``WikiTablesDecoderState``, except that we keep track of a
    checklist score, and other variables related to it.

    Parameters
    ----------
    agenda : ``List[torch.LongTensor]``
        List of agendas for instances, each of which is a tensor containing the indices of the
        actions we want to see in the decoded output
    agenda_mask : ``List[torch.LongTensor]``
        List of masks corresponding to agendas that indicate unpadded action items
    checklist : ``List[Variable]``
        A (soft) checklist for each instance indicating how many times each action in
        its agenda has been chosen previously. The checklist is soft because it contains the
        (sum of) the probabilities previously assigned to each action.
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
    """
    def __init__(self,
                 agenda: List[torch.LongTensor],
                 agenda_mask: List[torch.LongTensor],
                 checklist: List[Variable],
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
                 possible_actions: List[List[ProductionRuleArray]]) -> None:
        super(NlvrDecoderState, self).__init__(batch_indices, action_history, score)
        self.agenda = agenda
        self.agenda_mask = agenda_mask
        self.checklist = checklist
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

    def get_valid_actions(self) -> List[List[int]]:
        """
        This method is identical to ``WikiTablesDecoderState.get_valid_actions``.
        Returns a list of valid actions for each element of the group.
        """
        return [state.get_valid_actions() for state in self.grammar_state]

    def is_finished(self) -> bool:
        """This method is identical to ``WikiTablesDecoderState.is_finished``."""
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_state[0].is_finished()

    def split_finished(self) -> Tuple['NlvrDecoderState', 'NlvrDecoderState']:
        """This method is identical to ``WikiTablesDecoderState.split_finished``."""
        # We keep track of both of these so we can efficiently decide whether we need to split at
        # all.
        finished_indices = []
        not_finished_indices = []
        for i, state in enumerate(self.grammar_state):
            if state.is_finished():
                finished_indices.append(i)
            else:
                not_finished_indices.append(i)

        # Return value is (finished, not_finished)
        if not finished_indices:
            return (None, self)
        if not not_finished_indices:
            return (self, None)
        finished_state = self._make_new_state_with_group_indices(finished_indices)
        not_finished_state = self._make_new_state_with_group_indices(not_finished_indices)
        return (finished_state, not_finished_state)

    @classmethod
    def combine_states(cls, states) -> 'NlvrDecoderState':
        agenda = [agenda_list for state in states for agenda_list in state.agenda]
        agenda_mask = [mask_list for state in states for mask_list in state.agenda_mask]
        checklist = [checklist_list for state in states for checklist_list in state.checklist]
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        hidden_states = [hidden_state for state in states for hidden_state in state.hidden_state]
        memory_cells = [memory_cell for state in states for memory_cell in state.memory_cell]
        previous_action = [action for state in states for action in state.previous_action_embedding]
        attended_sentence = [attended for state in states for attended in state.attended_sentence]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        return NlvrDecoderState(agenda,
                                agenda_mask,
                                checklist,
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
                                states[0].possible_actions)

    def _make_new_state_with_group_indices(self, group_indices: List[int]) -> 'NlvrDecoderState':
        """
        This method's logic is identical to that of
        ``WikiTablesDecoderState._make_new_state_with_group_indices``.
        The ``NlvrDecoderState`` is `grouped`.  This is batching together the computation of
        many individual states, but we're using a different word here because it's not the same
        batching as the input training examples.  This method returns a new state that contains
        only selected elements of the group.  You might use this to split the group elements in a
        state into a finished state and a not finished state, for instance, if you know which group
        elements are finished.
        """
        group_agenda = [self.agenda[i] for i in group_indices]
        group_agenda_mask = [self.agenda_mask[i] for i in group_indices]
        group_checklist = [self.checklist[i] for i in group_indices]
        group_batch_indices = [self.batch_indices[i] for i in group_indices]
        group_action_histories = [self.action_history[i] for i in group_indices]
        group_scores = [self.score[i] for i in group_indices]
        group_previous_action = [self.previous_action_embedding[i] for i in group_indices]
        group_grammar_states = [self.grammar_state[i] for i in group_indices]
        group_hidden_states = [self.hidden_state[i] for i in group_indices]
        group_memory_cells = [self.memory_cell[i] for i in group_indices]
        group_attended_sentence = [self.attended_sentence[i] for i in group_indices]
        return NlvrDecoderState(group_agenda,
                                group_agenda_mask,
                                group_checklist,
                                group_batch_indices,
                                group_action_histories,
                                group_scores,
                                group_hidden_states,
                                group_memory_cells,
                                group_previous_action,
                                group_attended_sentence,
                                group_grammar_states,
                                self.encoder_outputs,
                                self.encoder_output_mask,
                                self.action_embeddings,
                                self.action_indices,
                                self.possible_actions)


class NlvrDecoderStep(DecoderStep[NlvrDecoderState]):
    def __init__(self,
                 encoder_output_dim: int,
                 action_embedding_dim: int,
                 attention_function: SimilarityFunction,
                 checklist_weight: float = 0.5) -> None:
        """
        Parameters
        ----------
        encoder_output_dim : ``int``
        action_embedding_dim : ``int``
        attention_function : ``SimilarityFunction``
        checklist_weight : ``float`` (optional)
            Weight assigned to checklist score when computing the final score. The final score will
            be ``checklist_weight * checklist_score + (1 - checklist_weight) * action_prob``.
        """
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
        # hidden state.  Then we concatenate that with the decoder state and project to
        # `action_embedding_dim` to make a prediction.
        self._output_projection_layer = Linear(output_dim + encoder_output_dim, action_embedding_dim)

        # TODO(pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(input_dim, output_dim)
        self._checklist_weight = checklist_weight

    @overrides
    def take_step(self,  # type: ignore
                  state: NlvrDecoderState,
                  max_actions: int = None) -> List[NlvrDecoderState]:
        """
        Given a ``NlvrDecoderState``, returns a list of next states that are sorted by their scores.
        This method is very similar to ``WikiTablesDecoderStep._take_step``. The differences are
        that we do not have a notion of "allowed actions" here, and we do not perform entity linking
        here.
        """
        # Outline here: first we'll construct the input to the decoder, which is a concatenation of
        # an embedding of the decoder input (the last action taken) and an attention over the
        # sentence.  Then we'll update our decoder's hidden state given this input, and recompute
        # an attention over the sentence given our new hidden state.  We'll use a concatenation of
        # the new hidden state and the new attention to predict an output, then yield new states.
        # Each new state corresponds to one valid action that can be taken from the current state,
        # and they are ordered by how much they contribute to an agenda.
        attended_sentence = torch.stack([x for x in state.attended_sentence])
        hidden_state = torch.stack([x for x in state.hidden_state])
        memory_cell = torch.stack([x for x in state.memory_cell])
        previous_action_embedding = torch.stack([x for x in state.previous_action_embedding])

        # (group_size, decoder_input_dim)
        decoder_input = self._input_projection_layer(torch.cat([attended_sentence,
                                                                previous_action_embedding], -1))

        hidden_state, memory_cell = self._decoder_cell(decoder_input, (hidden_state, memory_cell))

        # (group_size, encoder_output_dim)
        encoder_outputs = torch.stack([state.encoder_outputs[i] for i in state.batch_indices])
        encoder_output_mask = torch.stack([state.encoder_output_mask[i] for i in state.batch_indices])
        attended_sentence = self.attend_on_sentence(hidden_state, encoder_outputs,
                                                    encoder_output_mask)

        # To predict an action, we'll use a concatenation of the hidden state and attention over
        # the sentence.  We'll just predict an _embedding_, which we will compare to embedded
        # representations of all valid actions to get a final output.
        action_query = torch.cat([hidden_state, attended_sentence], dim=-1)

        # (group_size, action_embedding_dim)
        predicted_action_embedding = self._output_projection_layer(action_query)

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
        # We'll do a batch dot product here with `bmm`.  We want `dot(predicted_action_embedding,
        # action_embedding)` for each `action_embedding`, and we can get that efficiently with
        # `bmm` and some squeezing.
        # Shape: (group_size, num_embedded_actions)
        embedded_action_logits = action_embeddings.bmm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)

        action_logits = embedded_action_logits
        action_mask = embedded_action_mask.float()
        log_probs = nn_util.masked_log_softmax(action_logits, action_mask)

        return self._compute_new_states(state,
                                        log_probs,
                                        hidden_state,
                                        memory_cell,
                                        action_embeddings,
                                        attended_sentence,
                                        considered_actions,
                                        self._checklist_weight,
                                        max_actions)

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
        action_tensor = Variable(state.encoder_output_mask[0].data.new(padded_actions).long())
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
    def _compute_new_states(cls,  # type: ignore
                            state: NlvrDecoderState,
                            log_probs: torch.Tensor,
                            hidden_state: torch.Tensor,
                            memory_cell: torch.Tensor,
                            action_embeddings: torch.Tensor,
                            attended_sentence: torch.Tensor,
                            considered_actions: List[List[int]],
                            checklist_weight: float,
                            max_actions: int = None) -> List[NlvrDecoderState]:
        """
        This method is very similar to ``WikiTabledDecoderStep._compute_new_states``.
        The difference here is that we compute new states based on coverage loss.  For each action
        in considered actions, if it is in the agenda, we update the checklist
        and recalculate the score. We'll have to iterate through considered actions, but we have
        to do that to update action histories anyway.
        """
        # TODO(pradeep): We do not have a notion of ``allowed_actions`` for NLVR for now, but this
        # may be applicable in the future.
        # Note that we're not sorting these probs. We'll score checklists and sort states based on
        # checklist scores later.
        probs = torch.exp(log_probs)
        # batch_index -> [(group_index, action_index, action, checklist, score)]
        next_states_info: Dict[int, List[Tuple[int, int, int, Variable, Variable]]] = defaultdict(list)
        for group_index, (batch_index, instance_action_probs) in enumerate(zip(state.batch_indices, probs)):
            instance_agenda = state.agenda[group_index]  # (agenda_size,)
            instance_agenda_mask = state.agenda_mask[group_index]  # (agenda_size,)
            instance_checklist = state.checklist[group_index]  # (agenda_size,)
            # action_prob is a Variable.
            for action_index, action_prob in enumerate(instance_action_probs):
                # This is the actual index of the action from the original list of actions.
                action = considered_actions[group_index][action_index]
                if action == -1:
                    # Ignoring padding.
                    continue
                # If action is not in instance_agenda, mask_variable, and checklist_addition will be
                # all 0s.
                checklist_mask = instance_agenda == action  # (agenda_size,)
                checklist_addition = checklist_mask.float() * action_prob  # (agenda_size,)
                new_checklist = instance_checklist + checklist_addition  # (agenda_size,)
                checklist_score = cls.score_instance_checklist(new_checklist, instance_agenda_mask)
                new_score = checklist_weight * checklist_score + \
                            (1 - checklist_weight) * action_prob
                next_states_info[batch_index].append((group_index, action_index, action,
                                                      new_checklist, new_score))
        new_states = []
        for batch_index, states_info in next_states_info.items():
            # We need to sort states by scores. We batch all the scores first for efficient sorting.
            batch_scores = torch.cat([state_info[-1] for state_info in states_info])
            _, sorted_indices = batch_scores.sort(-1, descending=True)
            sorted_states_info = [states_info[i] for i in sorted_indices.cpu().data.numpy()]
            if max_actions is not None:
                sorted_states_info = sorted_states_info[:max_actions]
            for group_index, action_index, action, new_checklist, new_score in sorted_states_info:
                new_action_history = state.action_history[group_index] + [action]
                action_embedding = action_embeddings[group_index, action_index, :]
                left_side = state.possible_actions[batch_index][action]['left'][0]
                right_side = state.possible_actions[batch_index][action]['right'][0]
                new_grammar_state = state.grammar_state[group_index].take_action(left_side,
                                                                                 right_side)

                new_state = NlvrDecoderState([state.agenda[group_index]],
                                             [state.agenda_mask[group_index]],
                                             [new_checklist],
                                             [batch_index],
                                             [new_action_history],
                                             [new_score],
                                             [hidden_state[group_index]],
                                             [memory_cell[group_index]],
                                             [action_embedding],
                                             [attended_sentence[group_index]],
                                             [new_grammar_state],
                                             state.encoder_outputs,
                                             state.encoder_output_mask,
                                             state.action_embeddings,
                                             state.action_indices,
                                             state.possible_actions)
                new_states.append(new_state)
        return new_states

    @staticmethod
    def score_instance_checklist(checklist: Variable, agenda_mask: Variable) -> Variable:
        """
        Takes a checklist and agenda's mask (that shows which of the agenda items are not actually
        padding), and scores the checklist. We want each of the scores on the checklist to be as
        close to 1.0 as possible.
        """
        float_mask = agenda_mask.float()
        agenda_item_probs = checklist * float_mask
        score = -torch.sum((float_mask - agenda_item_probs) ** 2)
        return score
