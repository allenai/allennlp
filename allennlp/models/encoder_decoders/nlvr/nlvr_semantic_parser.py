import logging
from typing import Callable, List, Set, Dict, Tuple, Union

from overrides import overrides

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.decoding import DecoderTrainer, ExpectedRiskMinimization, RnnState
from allennlp.nn import util as nn_util
from allennlp.models.model import Model
from allennlp.models.encoder_decoders.nlvr.nlvr_decoder_state import NlvrDecoderState
from allennlp.models.encoder_decoders.nlvr.nlvr_decoder_step import NlvrDecoderStep
from allennlp.semparse.type_declarations.type_declaration import START_SYMBOL
from allennlp.semparse.type_declarations import GrammarState
from allennlp.semparse.worlds import NlvrWorld
from allennlp.training.metrics import Average

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("nlvr_parser")
class NlvrSemanticParser(Model):
    """
    ``NlvrSemanticParser`` is a semantic parsing model built for the NLVR domain.  There is a lot of
    overlap with ``WikiTablesSemanticParser`` here. We may want to eventually move the common
    functionality into a more general transition-based parsing class.

    The main differences between this parser and what we have for Wikitables are that we have an
    agenda of actions instead of complete target action sequences, and accordingly the score in this
    parser is based on how many of the agenda actions are covered.

    Parameters
    ----------
    vocab : ``Vocabulary``
    sentence_embedder : ``TextFieldEmbedder``
        Embedder for sentences.
    nonterminal_embedder : ``TextFieldEmbedder``
        We will embed nonterminals in the grammar using this embedder.  These aren't
        ``TextFields``, but they are structured the same way.
    terminal_embedder : ``TextFieldEmbedder``
        We will embed terminals in the grammar using this embedder.  These aren't ``TextFields``,
        but they are structured the same way.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input question.
    attention_function : ``SimilarityFunction``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  This is the similarity function we use for that
        attention.
    beam_size : ``int``
    normalize_beam_score_by_length : ``bool``, optional (default=False)
        Should the log probabilities be normalized by length before renormalizing them? Edunov et
        al. do this in their work, but we found that not doing it works better. It's possible they
        did this because their task is NMT, and longer decoded sequences are not necessarily worse,
        and shouldn't be penalized, while we will mostly want to penalize longer logical forms.
    max_decoding_steps : ``int``
        We use a beam search during decoding; what's the maximum number of steps we should take?
    checklist_cost_weight : ``float``, optional (default=0.8)
        Mixture weight (0-1) for combining coverage cost and denotation cost. As this increases, we
        weigh the coverage cost higher, with a value of 1.0 meaning that we do not care about
        denotation accuracy.
    dynamic_cost_weight : ``Dict[str, Union[int, float]]``, optional (default=None)
        A dict containing keys ``wait_num_epochs`` and ``rate`` indicating the number of steps
        after which we should start decreasing the weight on checklist cost in favor of denotation
        cost, and the rate at which we should do it. We will decrease the weight in the following
        way - ``checklist_cost_weight = checklist_cost_weight - rate * checklist_cost_weight``
        starting at the apropriate epoch.  The weight will remain constant if this is not provided.
    penalize_non_agenda_actions : ``bool``, optional (default=False)
        Should we penalize the model for producing terminal actions that are outside the agenda?
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 nonterminal_embedder: TextFieldEmbedder,
                 terminal_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention_function: SimilarityFunction,
                 beam_size: int,
                 max_decoding_steps: int,
                 normalize_beam_score_by_length: bool = False,
                 checklist_cost_weight: float = 0.8,
                 dynamic_cost_weight: Dict[str, Union[int, float]] = None,
                 penalize_non_agenda_actions: bool = False) -> None:
        super(NlvrSemanticParser, self).__init__(vocab=vocab)

        self._sentence_embedder = sentence_embedder
        self._denotation_accuracy = Average()
        self._agenda_coverage = Average()
        check_dimensions_match(nonterminal_embedder.get_output_dim(),
                               terminal_embedder.get_output_dim(),
                               "nonterminal embedding dim",
                               "terminal embedding dim")
        self._nonterminal_embedder = nonterminal_embedder
        self._terminal_embedder = terminal_embedder
        self._encoder = encoder
        self._decoder_trainer: DecoderTrainer[Callable[[NlvrDecoderState], torch.Tensor]] = \
                ExpectedRiskMinimization(beam_size, normalize_beam_score_by_length, max_decoding_steps)
        action_embedding_dim = nonterminal_embedder.get_output_dim() * 2

        # Instantiating an empty NlvrWorld just to get the number of terminals.
        num_terminals = len(NlvrWorld([]).terminal_productions)
        self._decoder_step = NlvrDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                             action_embedding_dim=action_embedding_dim,
                                             attention_function=attention_function,
                                             checklist_size=num_terminals)
        self._checklist_cost_weight = checklist_cost_weight
        self._dynamic_cost_wait_epochs = None
        self._dynamic_cost_rate = None
        if dynamic_cost_weight:
            self._dynamic_cost_wait_epochs = dynamic_cost_weight["wait_num_epochs"]
            self._dynamic_cost_rate = dynamic_cost_weight["rate"]
        self._penalize_non_agenda_actions = penalize_non_agenda_actions
        self._last_epoch_in_forward: int = None

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                world: List[NlvrWorld],
                actions: List[List[ProductionRuleArray]],
                agenda: torch.LongTensor,
                label: torch.LongTensor = None,
                epoch_num: List[int] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing type constrained target sequences that maximize coverage of
        their respective agendas, and minimize a denotation based loss.
        """
        # We look at the epoch number and adjust the checklist cost weight if needed here.
        instance_epoch_num = epoch_num[0] if epoch_num is not None else None
        if self._dynamic_cost_rate is not None:
            if instance_epoch_num is None:
                raise RuntimeError("If you want a dynamic cost weight, use the "
                                   "EpochTrackingBucketIterator!")
            if instance_epoch_num != self._last_epoch_in_forward:
                if instance_epoch_num >= self._dynamic_cost_wait_epochs:
                    decrement = self._checklist_cost_weight * self._dynamic_cost_rate
                    self._checklist_cost_weight -= decrement
                    logger.info("Checklist cost weight is now %f", self._checklist_cost_weight)
                self._last_epoch_in_forward = instance_epoch_num

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
        encoder_output_dim = encoder_outputs.size(2)
        # Expanding indices to 3 dimensions
        expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
        # (batch_size, 1, encoder_output_dim)
        final_encoder_output = encoder_outputs.gather(1, expanded_indices)
        final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
        memory_cell = nn_util.new_variable_with_size(encoder_outputs,
                                                     (batch_size, self._encoder.get_output_dim()),
                                                     0)
        action_embeddings, action_indices, initial_action_embedding = self._embed_actions(actions)
        # TODO (pradeep): Use an unindexed field for labels?
        labels_data = label.data.cpu()
        label_strings = [self.vocab.get_token_from_index(int(label_data), "denotations") for
                         label_data in labels_data]
        # Each instance's agenda is of size (agenda_size, 1)
        agenda_list = [agenda[i] for i in range(batch_size)]
        checklist_targets = []
        all_terminal_actions = []
        checklist_masks = []
        initial_checklist_list = []
        for instance_actions, instance_agenda in zip(actions, agenda_list):
            checklist_info = self._get_checklist_info(instance_agenda, instance_actions)
            checklist_target, terminal_actions, checklist_mask = checklist_info
            checklist_targets.append(checklist_target)
            all_terminal_actions.append(terminal_actions)
            checklist_masks.append(checklist_mask)
            initial_checklist_list.append(nn_util.new_variable_with_size(checklist_target,
                                                                         checklist_target.size(),
                                                                         0))
        attended_sentence = self._decoder_step.attend_on_sentence(final_encoder_output,
                                                                  encoder_outputs, sentence_mask)
        initial_score_list = [nn_util.new_variable_with_data(agenda, torch.Tensor([0.0])) for i in
                              range(batch_size)]
        encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
        sentence_mask_list = [sentence_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnState(final_encoder_output[i],
                                              memory_cell[i],
                                              initial_action_embedding,
                                              attended_sentence[i],
                                              encoder_outputs_list,
                                              sentence_mask_list))

        initial_grammar_state = [self._create_grammar_state(world[i], actions[i]) for i in
                                 range(batch_size)]
        worlds_list = [world[i] for i in range(batch_size)]
        initial_state = NlvrDecoderState(batch_indices=list(range(batch_size)),
                                         action_history=[[] for _ in range(batch_size)],
                                         score=initial_score_list,
                                         rnn_state=initial_rnn_state,
                                         grammar_state=initial_grammar_state,
                                         terminal_actions=all_terminal_actions,
                                         checklist_target=checklist_targets,
                                         checklist_masks=checklist_masks,
                                         checklist=initial_checklist_list,
                                         action_embeddings=action_embeddings,
                                         action_indices=action_indices,
                                         possible_actions=actions,
                                         worlds=worlds_list,
                                         label_strings=label_strings)

        outputs = self._decoder_trainer.decode(initial_state, self._decoder_step, self._get_state_cost)
        agenda_data = [agenda_[:, 0].cpu().data for agenda_ in agenda_list]
        best_action_sequences = outputs['best_action_sequence']
        get_action_string = lambda rule: "%s -> %s" % (rule["left"][0], rule["right"][0])
        for i in range(batch_size):
            batch_actions = actions[i]
            batch_best_sequences = best_action_sequences[i] if i in best_action_sequences else []
            sequence_is_correct = False
            in_agenda_ratio = 0.0
            if batch_best_sequences:
                action_strings = [get_action_string(batch_actions[rule_id]) for rule_id in
                                  batch_best_sequences]
                terminal_agenda_actions = []
                for rule_id in agenda_data[i]:
                    if rule_id == -1:
                        continue
                    action_string = get_action_string(batch_actions[rule_id])
                    right_side = action_string.split(" -> ")[1]
                    if right_side.isdigit() or ('[' not in right_side and len(right_side) > 1):
                        terminal_agenda_actions.append(rule_id)
                actions_in_agenda = [rule_id in batch_best_sequences for rule_id in
                                     terminal_agenda_actions]
                in_agenda_ratio = sum(actions_in_agenda) / len(actions_in_agenda)
                label_string = label_strings[i]
                instance_world = world[i]
                sequence_is_correct = self._check_denotation(action_strings,
                                                             label_string,
                                                             instance_world)
            self._denotation_accuracy(1 if sequence_is_correct else 0)
            self._agenda_coverage(in_agenda_ratio)
        return outputs

    def _get_checklist_info(self,
                            agenda: torch.LongTensor,
                            all_actions: List[ProductionRuleArray]) -> Tuple[torch.Tensor,
                                                                             torch.Tensor,
                                                                             torch.Tensor]:
        """
        Takes an agenda and a list of all actions and returns a target checklist against which the
        checklist at each state will be compared to compute a loss, indices of ``terminal_actions``,
        and a ``checklist_mask`` that indicates which of the terminal actions are relevant for
        checklist loss computation. If ``self.penalize_non_agenda_actions`` is set to``True``,
        ``checklist_mask`` will be all 1s (i.e., all terminal actions are relevant). If it is set to
        ``False``, indices of all terminals that are not in the agenda will be masked.

        Parameters
        ----------
        ``agenda`` : ``torch.LongTensor``
            Agenda of one instance of size ``(agenda_size, 1)``.
        ``all_actions`` : ``List[ProductionRuleArray]``
            All actions for one instance.
        """
        terminal_indices = []
        target_checklist_list = []
        agenda_indices_set = set([int(x) for x in agenda.squeeze(0).data.cpu().numpy()])
        for index, action in enumerate(all_actions):
            # Each action is a ProductionRuleArray, a dict with keys "left" and "right", and
            # values are tuples where the second element shows whether element is a
            # non_terminal.
            if not action["right"][1]:
                terminal_indices.append([index])
                if index in agenda_indices_set:
                    target_checklist_list.append([1])
                else:
                    target_checklist_list.append([0])
        # We want to return checklist target and terminal actions that are column vectors to make
        # computing softmax over the difference between checklist and target easier.
        # (num_terminals, 1)
        terminal_actions = nn_util.new_variable_with_data(agenda,
                                                          torch.Tensor(terminal_indices))
        # (num_terminals, 1)
        target_checklist = nn_util.new_variable_with_data(agenda,
                                                          torch.Tensor(target_checklist_list))
        if self._penalize_non_agenda_actions:
            # All terminal actions are relevant
            checklist_mask = torch.ones_like(target_checklist)
        else:
            checklist_mask = (target_checklist != 0).float()
        return target_checklist, terminal_actions, checklist_mask

    @staticmethod
    def _check_denotation(best_action_sequence: List[str],
                          label: str,
                          world: NlvrWorld) -> bool:
        logical_form = world.get_logical_form(best_action_sequence)
        denotation = world.execute(logical_form)
        is_correct = str(denotation).lower() == label.lower()
        return is_correct

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'denotation_accuracy': self._denotation_accuracy.get_metric(reset),
                'agenda_coverage': self._agenda_coverage.get_metric(reset)
        }

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

    def _get_state_cost(self, state: NlvrDecoderState) -> torch.Tensor:
        """
        Return the costs a finished state. Since it is a finished state, the group size will be 1,
        and hence we'll return just one cost.
        """
        if not state.is_finished():
            raise RuntimeError("_get_state_cost() is not defined for unfinished states!")
        instance_checklist_target = state.checklist_target[0]
        instance_checklist = state.checklist[0]
        instance_checklist_mask = state.checklist_mask[0]

        # Our checklist cost is a sum of squared error from where we want to be, making sure we
        # take into account the mask.
        checklist_balance = instance_checklist_target - instance_checklist
        checklist_balance = checklist_balance * instance_checklist_mask
        checklist_cost = torch.sum((checklist_balance) ** 2)

        # This is the number of items on the agenda that we want to see in the decoded sequence.
        # We use this as the denotation cost if the path is incorrect.
        # Note: If we are penalizing the model for producing non-agenda actions, this is not the
        # upper limit on the checklist cost. That would be the number of terminal actions.
        denotation_cost = torch.sum(instance_checklist_target.float())
        checklist_cost = self._checklist_cost_weight * checklist_cost
        if self._denotation_is_correct(state):
            cost = checklist_cost
        else:
            cost = checklist_cost + (1 - self._checklist_cost_weight) * denotation_cost
        return cost

    def _denotation_is_correct(self, state) -> bool:
        """
        Returns whether action history in the state evaluates to the correct denotation. Only
        defined when the state is finished.
        """
        assert state.is_finished(), "Cannot compute denotations for unfinished states!"
        # Since this is a finished state, its group size must be 1.
        batch_index = state.batch_indices[0]
        world = state.worlds[batch_index]
        label_string = state.label_strings[batch_index]
        history = state.action_history[0]
        all_actions = state.possible_actions[0]
        action_sequence = [self._get_action_string(all_actions[action]) for action in history]
        logical_form = world.get_logical_form(action_sequence)
        denotation = world.execute(logical_form)
        is_correct = str(denotation).lower() == label_string.lower()
        return is_correct

    def _get_state_info(self, state) -> Dict[str, List]:
        """
        This method is here for debugging purposes, in case you want to look at the what the model
        is learning. It may be inefficient to call it while training the model on real data.
        """
        if len(state.batch_indices) == 1 and state.is_finished():
            costs = [float(self._get_state_cost(state).data.cpu().numpy())]
        else:
            costs = []
        model_scores = [float(score.data.cpu().numpy()) for score in state.score]
        all_actions = state.possible_actions[0]
        action_sequences = [[self._get_action_string(all_actions[action]) for action in history]
                            for history in state.action_history]
        agenda_sequences = []
        all_agenda_indices = []
        for agenda, checklist_target in zip(state.terminal_actions, state.checklist_target):
            agenda_indices = []
            for action, is_wanted in zip(agenda, checklist_target):
                action_int = int(action.data.cpu().numpy())
                is_wanted_int = int(is_wanted.data.cpu().numpy())
                if is_wanted_int != 0:
                    agenda_indices.append(action_int)
            agenda_sequences.append([self._get_action_string(all_actions[action])
                                     for action in agenda_indices])
            all_agenda_indices.append(agenda_indices)
        return {"agenda": agenda_sequences,
                "agenda_indices": all_agenda_indices,
                "history": action_sequences,
                "history_indices": state.action_history,
                "costs": costs,
                "scores": model_scores}

    @staticmethod
    def _get_action_string(production_rule: ProductionRuleArray) -> str:
        return "%s -> %s" % (production_rule["left"][0], production_rule["right"][0])


    @classmethod
    def from_params(cls, vocab, params: Params) -> 'NlvrSemanticParser':
        sentence_embedder_params = params.pop("sentence_embedder")
        sentence_embedder = TextFieldEmbedder.from_params(vocab, sentence_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        nonterminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("nonterminal_embedder"))
        terminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("terminal_embedder"))
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        beam_size = params.pop_int('beam_size')
        normalize_beam_score_by_length = params.pop_bool('normalize_beam_score_by_length', False)
        max_decoding_steps = params.pop_int("max_decoding_steps")
        checklist_cost_weight = params.pop_float("checklist_cost_weight", 0.8)
        dynamic_cost_weight = params.pop("dynamic_cost_weight", None)
        penalize_non_agenda_actions = params.pop_bool("penalize_non_agenda_actions", False)
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   sentence_embedder=sentence_embedder,
                   nonterminal_embedder=nonterminal_embedder,
                   terminal_embedder=terminal_embedder,
                   encoder=encoder,
                   attention_function=attention_function,
                   beam_size=beam_size,
                   max_decoding_steps=max_decoding_steps,
                   normalize_beam_score_by_length=normalize_beam_score_by_length,
                   checklist_cost_weight=checklist_cost_weight,
                   dynamic_cost_weight=dynamic_cost_weight,
                   penalize_non_agenda_actions=penalize_non_agenda_actions)
