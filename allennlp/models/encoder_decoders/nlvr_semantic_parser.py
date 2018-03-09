# pylint: disable=too-many-lines
from typing import List, Set, Dict, Tuple
from collections import defaultdict

from overrides import overrides

import numpy
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
    decoder_trainer : ``DecoderTrainer``
        The structured learning algorithm used to train the decoder (which also trains the encoder,
        but it's applied to the decoder outputs).
    max_decoding_steps : ``int``
        Maximum number of decoding steps used for training.
    attention_function : ``SimilarityFunction``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  This is the similarity function we use for that
        attention.
    checklist_selection_weight : ``float``
        The mixture weight for combining model probabilities for all actions and probabilities for
        those in agenda based on how much they contribute to the checklist.
    checklist_cost_weight : ``float``
        Mixture weight (0-1) for combining coverage cost and denotation cost. As this increases, we
        weigh the coverage cost higher, with a value of 1.0 meaning that we do not care about
        denotation accuracy.
    penalize_non_agenda_actions : ``bool``
        Should we penalize the model for producing terminal actions that are outside the agenda?
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 nonterminal_embedder: TextFieldEmbedder,
                 terminal_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder_trainer: DecoderTrainer,
                 max_decoding_steps: int,
                 attention_function: SimilarityFunction,
                 checklist_selection_weight: float,
                 checklist_cost_weight: float,
                 penalize_non_agenda_actions: bool) -> None:
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
        self._decoder_trainer = decoder_trainer
        self._max_decoding_steps = max_decoding_steps
        self._attention_function = attention_function
        action_embedding_dim = nonterminal_embedder.get_output_dim() * 2

        self._decoder_step = NlvrDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                             action_embedding_dim=action_embedding_dim,
                                             attention_function=attention_function,
                                             checklist_weight=checklist_selection_weight)
        self._checklist_cost_weight = checklist_cost_weight
        self._penalize_non_agenda_actions = penalize_non_agenda_actions

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                world: List[NlvrWorld],
                actions: List[List[ProductionRuleArray]],
                agenda: torch.LongTensor,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing type constrained target sequences, that maximize coverage of
        their respective agendas. This will change soon, to include a denotation based score as
        well, once we have a way to transform action sequences into logical forms that can be
        executed to produce denotations.
        """
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
        # TODO (pradeep): Use an unindexed field for labels?
        labels_data = label.data.cpu()
        label_strings = [self.vocab.get_token_from_index(int(label_data), "denotations") for
                         label_data in labels_data]
        # Each instance's agenda is of size (agenda_size, 1)
        agenda_list = [agenda[i] for i in range(batch_size)]
        checklist_targets = []
        agenda_relevant_actions = []
        initial_checklist_list = []
        for instance_actions, instance_agenda in zip(actions, agenda_list):
            checklist_target, relevant_actions = self._get_checklist_target(instance_agenda,
                                                                            instance_actions)
            checklist_targets.append(checklist_target)
            agenda_relevant_actions.append(relevant_actions)
            initial_checklist_list.append(nn_util.new_variable_with_size(checklist_target,
                                                                         checklist_target.size(),
                                                                         0))
        initial_score_list = [nn_util.new_variable_with_data(agenda, torch.Tensor([0.0])) for i in
                              range(batch_size)]
        initial_hidden_state = [final_encoder_output[i] for i in range(batch_size)]
        initial_memory_cell = [memory_cell[i] for i in range(batch_size)]
        initial_action_embedding_list = [initial_action_embedding for _ in range(batch_size)]
        initial_grammar_state = [self._create_grammar_state(world[i], actions[i]) for i in
                                 range(batch_size)]
        initial_attended_sentence = [attended_sentence[i] for i in range(batch_size)]
        encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
        sentence_mask_list = [sentence_mask[i] for i in range(batch_size)]
        worlds_list = [world[i] for i in range(batch_size)]
        initial_state = NlvrDecoderState(agenda_relevant_actions,
                                         checklist_targets,
                                         initial_checklist_list,
                                         self._checklist_cost_weight,
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
                                         actions,
                                         worlds_list,
                                         label_strings)

        outputs = self._decoder_trainer.decode(initial_state, self._decoder_step,  # type: ignore
                                               self._max_decoding_steps)
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

    def _get_checklist_target(self,
                              agenda: torch.LongTensor,
                              all_actions: List[ProductionRuleArray]) -> Tuple[torch.LongTensor,
                                                                               torch.LongTensor]:
        """
        Takes an agenda and a list of all actions and returns a target checklist against which the
        checklist at each state will be compared against to compute a loss, and the indices of
        actions relevant for checklist loss computation (``relevant_actions``). If
        ``penalize_non_agenda_actions`` is set to ``True``, ``relevant_actions`` will contain
        indices to all the terminal actions. If not, we will simply return the ``agenda`` itself as
        ``relevant_actions``, in which case, it will be a padded tensor, with -1 indicating padding.

        Parameters
        ----------
        ``agenda`` : ``torch.LongTensor``
            Agenda of one instance of size ``(agenda_size, 1)``.
        ``all_actions`` : ``List[ProductionRuleArray]``
            All actions for one instance.
        """
        if self._penalize_non_agenda_actions:
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
            # We want to return a checklist target and the relevant actions that are column vectors to
            # make computing softmax over the difference between checklist and target easier.
            # (num_terminals, 1)
            relevant_actions = nn_util.new_variable_with_data(agenda,
                                                              torch.Tensor(terminal_indices))
            # (num_terminals, 1)
            target_checklist = nn_util.new_variable_with_data(agenda,
                                                              torch.Tensor(target_checklist_list))
        else:
            relevant_actions = agenda  # (agenda_size, 1)
            target_checklist = (agenda != -1).float()  # (agenda_size, 1)
        return target_checklist, relevant_actions


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
        checklist_selection_weight = params.pop_float("checklist_selection_weight", 0.5)
        checklist_cost_weight = params.pop_float("checklist_cost_weight", 0.8)
        penalize_non_agenda_actions = params.pop_bool("penalize_non_agenda_actions", False)
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   sentence_embedder=sentence_embedder,
                   nonterminal_embedder=nonterminal_embedder,
                   terminal_embedder=terminal_embedder,
                   encoder=encoder,
                   decoder_trainer=decoder_trainer,
                   max_decoding_steps=max_decoding_steps,
                   attention_function=attention_function,
                   checklist_selection_weight=checklist_selection_weight,
                   checklist_cost_weight=checklist_cost_weight,
                   penalize_non_agenda_actions=penalize_non_agenda_actions)


class NlvrDecoderState(DecoderState['NlvrDecoderState']):
    """
    This class is very similar to ``WikiTablesDecoderState``, except that we keep track of a
    checklist score, and other variables related to it.

    Parameters
    ----------
    agenda_relevant_actions : ``List[torch.LongTensor]``
        List of actions relevant for computing checklist costs for instances, each of which is a
        tensor containing the indices of the actions we want to see or not see in the decoded
        output
    checklist_target : ``List[torch.LongTensor]``
        List of targets corresponding to agendas that indicate the states we want the checklists to
        ideally be. Each element in this list is the same size as the corresponding element in
        ``agenda_relevant_actions``, and it contains 1 for each corresponding action in the relevant
        actions list that we want to see in the final logical form, and 0 for each corresponding
        action that we do not.
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
                 agenda_relevant_actions: List[torch.LongTensor],
                 checklist_target: List[torch.LongTensor],
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
        self.agenda_relevant_actions = agenda_relevant_actions
        self.checklist_target = checklist_target
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

    def get_valid_actions(self, shuffle: bool = True) -> List[List[int]]:
        """
        This method is identical to ``WikiTablesDecoderState.get_valid_actions``.
        Returns a list of valid actions for each element of the group.
        """
        valid_actions = [state.get_valid_actions() for state in self.grammar_state]
        if shuffle:
            for actions in valid_actions:
                # In-place shuffle
                numpy.random.shuffle(actions)
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
        for agenda, checklist_target in zip(self.agenda_relevant_actions, self.checklist_target):
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


    def get_agenda_action_probs(self) -> List[Variable]:
        """
        Returns probabilities of choosing in-agenda actions based on how many times they've been
        chosen previously. That is, if the checklist value for the first action is higher than that
        for the second action, the returned probability for the second action will be higher than
        that for the first action.
        """
        agenda_probs: List[Variable] = []
        for instance_target, instance_checklist in zip(self.checklist_target, self.checklist):
            checklist_balance = self._get_checklist_balance(instance_target, instance_checklist)
            # This assigns probabilities uniformly to all previously unselected actions. That is, if
            # the checklist_balance is [0, 1, 1, 0], we return [0, 0.5, 0.5, 0]; if it is [0, 1, 1,
            # 1], we return [0, 0.33, 0.33, 0.33], so on.
            agenda_probs.append(nn_util.masked_softmax(checklist_balance,
                                                       (checklist_balance != 0).float()))
        return agenda_probs

    @staticmethod
    def _get_checklist_balance(target, checklist):
        """
        This returns a float vector containing just 1s and 0s showing which of the items are
        filled. We clamp the min at 0 to ignore the number of times an action is taken. The value
        at an index will be 1 iff the target wants an action to be taken, and it is not yet taken.
        """
        checklist_balance = torch.clamp(target - checklist, min=0.0)
        return checklist_balance

    def get_cost(self) -> Variable:
        """
        Return the costs a finished state. Since it is a finished state, the group size will be 1,
        and hence we'll return just one cost.
        """
        if not self.is_finished():
            raise RuntimeError("get_costs() is not defined for unfinished states!")
        instance_checklist_target = self.checklist_target[0]
        instance_checklist = self.checklist[0]
        checklist_cost = - self.score_single_checklist(instance_checklist, instance_checklist_target)
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
                               instance_checklist_target: Variable) -> Variable:
        """
        Takes a single checklist and a corresponding checklist target and returns
        the score of the checklist. We want the checklist to be as close to the target as possible.
        """
        return -torch.sum((instance_checklist_target - instance_checklist) ** 2)

    def is_finished(self) -> bool:
        """This method is identical to ``WikiTablesDecoderState.is_finished``."""
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_state[0].is_finished()

    @classmethod
    def combine_states(cls, states) -> 'NlvrDecoderState':
        relevant_actions = [actions for state in states for actions in state.agenda_relevant_actions]
        checklist_target = [target_list for state in states for target_list in
                            state.checklist_target]
        checklist = [checklist_list for state in states for checklist_list in state.checklist]
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        hidden_states = [hidden_state for state in states for hidden_state in state.hidden_state]
        memory_cells = [memory_cell for state in states for memory_cell in state.memory_cell]
        previous_action = [action for state in states for action in state.previous_action_embedding]
        attended_sentence = [attended for state in states for attended in state.attended_sentence]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        return NlvrDecoderState(relevant_actions,
                                checklist_target,
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


class NlvrDecoderStep(DecoderStep[NlvrDecoderState]):
    def __init__(self,
                 encoder_output_dim: int,
                 action_embedding_dim: int,
                 attention_function: SimilarityFunction,
                 checklist_weight: float) -> None:
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
        # TODO(pradeep): Break this method into smaller methods.
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
        considered_action_probs = nn_util.masked_softmax(action_logits, action_mask)
        all_agenda_action_probs = state.get_agenda_action_probs()
        # Mixing model scores and agenda selection probabilities to compute the probabilities of all
        # actions for the next step and the corresponding new checklists.
        # All action logprobs will keep track of logprob corresponding to each local action index
        # for each instance.
        all_action_logprobs: List[List[Tuple[int, torch.Tensor]]] = []
        all_new_checklists: List[List[torch.LongTensor]] = []
        for group_index, instance_info in enumerate(zip(state.batch_indices,
                                                        state.score,
                                                        considered_action_probs,
                                                        state.checklist)):
            (batch_index, instance_score, instance_probs, instance_checklist) = instance_info
            instance_agenda = state.agenda_relevant_actions[group_index]  # (agenda_size, 1)
            agenda_action_probs = all_agenda_action_probs[group_index]
            # We will mix the model scores with agenda selection probabilities and compute their
            # logs to fill the following list with action indices and corresponding logprobs.
            instance_action_logprobs: List[Tuple[int, torch.Tensor]] = []
            instance_new_checklists: List[torch.LongTensor] = []
            for action_index, action_prob in enumerate(instance_probs):
                # This is the actual index of the action from the original list of actions.
                action = considered_actions[group_index][action_index]
                if action == -1:
                    # Ignoring padding.
                    continue
                # If action is not in instance_agenda, mask_variable, and checklist_addition will be
                # all 0s.
                checklist_mask = (instance_agenda == action).float()  # (agenda_size, 1)
                agenda_action_prob = torch.sum(checklist_mask * agenda_action_probs)  # (1,)
                # TODO (pradeep): This is not a great way to bias the model towards choosing actions
                # that fill the checklist. May be use the current checklist in the action logit
                # computation itself.
                action_prob = self._checklist_weight * agenda_action_prob + \
                              (1 - self._checklist_weight) * action_prob
                # We're adding 1.0 at the corresponding agenda index.
                checklist_addition = checklist_mask.float()  # (agenda_size, 1)
                new_checklist = instance_checklist + checklist_addition  # (agenda_size, 1)
                instance_new_checklists.append(new_checklist)
                logprob = instance_score + torch.log(action_prob + 1e-13)
                instance_action_logprobs.append((action_index, logprob))
            all_action_logprobs.append(instance_action_logprobs)
            all_new_checklists.append(instance_new_checklists)
        return self._compute_new_states(state,
                                        all_action_logprobs,
                                        all_new_checklists,
                                        hidden_state,
                                        memory_cell,
                                        action_embeddings,
                                        attended_sentence,
                                        considered_actions,
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
                            action_logprobs: List[List[Tuple[int, torch.Tensor]]],
                            new_checklists: List[List[torch.Tensor]],
                            hidden_state: torch.Tensor,
                            memory_cell: torch.Tensor,
                            action_embeddings: torch.Tensor,
                            attended_sentence: torch.Tensor,
                            considered_actions: List[List[int]],
                            max_actions: int = None) -> List[NlvrDecoderState]:
        """
        This method is very similar to ``WikiTabledDecoderStep._compute_new_states``.
        The difference here is that we also have checklists to deal with.
        """
        # TODO(pradeep): We do not have a notion of ``allowed_actions`` for NLVR for now, but this
        # may be applicable in the future.
        # batch_index -> group_index, action_index, checklist, score
        states_info: Dict[int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for group_index, instance_info in enumerate(zip(state.batch_indices,
                                                        action_logprobs,
                                                        new_checklists)):
            batch_index, instance_logprobs, instance_new_checklists = instance_info
            for (action_index, score), checklist in zip(instance_logprobs, instance_new_checklists):
                states_info[batch_index].append((group_index, action_index, checklist, score))

        new_states = []
        for batch_index, instance_states_info in states_info.items():
            batch_scores = torch.cat([info[-1] for info in instance_states_info])
            _, sorted_indices = batch_scores.sort(-1, descending=True)
            sorted_states_info = [instance_states_info[i] for i in sorted_indices.data.cpu().numpy()]
            if max_actions is not None:
                sorted_states_info = sorted_states_info[:max_actions]
            for group_index, action_index, new_checklist, new_score in sorted_states_info:
                # This is the actual index of the action from the original list of actions.
                # We do not have to check whether it is the padding index because ``take_step``
                # already took care of that.
                action = considered_actions[group_index][action_index]
                action_embedding = action_embeddings[group_index, action_index, :]
                new_action_history = state.action_history[group_index] + [action]
                left_side = state.possible_actions[batch_index][action]['left'][0]
                right_side = state.possible_actions[batch_index][action]['right'][0]
                new_grammar_state = state.grammar_state[group_index].take_action(left_side,
                                                                                 right_side)

                new_state = NlvrDecoderState([state.agenda_relevant_actions[group_index]],
                                             [state.checklist_target[group_index]],
                                             [new_checklist],
                                             state.checklist_cost_weight,
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
                                             state.possible_actions,
                                             state.worlds,
                                             state.label_strings)
                new_states.append(new_state)
        return new_states
