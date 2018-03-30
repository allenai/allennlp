import logging
from typing import List, Set, Dict, Tuple

from overrides import overrides

import torch
from torch.autograd import Variable

from allennlp.common.checks import check_dimensions_match
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.decoding import RnnState
from allennlp.nn import util as nn_util
from allennlp.models.model import Model
from allennlp.semparse.type_declarations.type_declaration import START_SYMBOL
from allennlp.semparse.type_declarations import GrammarState
from allennlp.semparse.worlds import NlvrWorld
from allennlp.training.metrics import Average

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NlvrSemanticParser(Model):
    """
    ``NlvrSemanticParser`` is a semantic parsing model built for the NLVR domain. This is an
    abstract class and does not have a ``forward`` method implemented. Classes that inherit from
    this class are expected to define their own logic depending on the kind of supervision they use.
    Accordingly, they should use the appropriate ``DecoderTrainer``. This class provides some common
    functionality for things like defining an initial ``RnnState``, embedding actions, evaluating
    the denotations of completed logical forms, etc.  There is a lot of overlap with
    ``WikiTablesSemanticParser`` here. We may want to eventually move the common functionality into
    a more general transition-based parsing class.

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
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 nonterminal_embedder: TextFieldEmbedder,
                 terminal_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super(NlvrSemanticParser, self).__init__(vocab=vocab)

        self._sentence_embedder = sentence_embedder
        self._denotation_accuracy = Average()
        self._consistency = Average()
        check_dimensions_match(nonterminal_embedder.get_output_dim(),
                               terminal_embedder.get_output_dim(),
                               "nonterminal embedding dim",
                               "terminal embedding dim")
        self._nonterminal_embedder = nonterminal_embedder
        self._terminal_embedder = terminal_embedder
        self._encoder = encoder

    @overrides
    def forward(self):  # type: ignore
        # pylint: disable=arguments-differ
        # Sub-classes should define their own logic here.
        raise NotImplementedError

    def _get_initial_rnn_state(self, sentence, initial_action_embedding):
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
        attended_sentence = self._decoder_step.attend_on_sentence(final_encoder_output,
                                                                  encoder_outputs, sentence_mask)
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
        return initial_rnn_state

    def _get_label_strings(self, labels):
        # TODO (pradeep): Use an unindexed field for labels?
        labels_data = labels.data.cpu()
        label_strings: List[List[str]] = []
        for instance_labels_data in labels_data:
            label_strings.append([])
            for label in instance_labels_data:
                label_int = int(label)
                if label_int == -1:
                    # Padding, because not all instances have the same number of labels.
                    continue
                label_strings[-1].append(self.vocab.get_token_from_index(label_int, "denotations"))
        return label_strings

    @staticmethod
    def _check_denotation(best_action_sequence: List[str],
                          labels: List[str],
                          worlds: List[NlvrWorld]) -> List[bool]:
        is_correct = []
        for world, label in zip(worlds, labels):
            logical_form = world.get_logical_form(best_action_sequence)
            denotation = world.execute(logical_form)
            is_correct.append(str(denotation).lower() == label)
        return is_correct

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

    def _check_state_denotations(self, state) -> List[bool]:
        """
        Returns whether action history in the state evaluates to the correct denotations over all
        worlds. Only defined when the state is finished.
        """
        assert state.is_finished(), "Cannot compute denotations for unfinished states!"
        # Since this is a finished state, its group size must be 1.
        batch_index = state.batch_indices[0]
        worlds = state.worlds[batch_index]
        instance_label_strings = state.label_strings[batch_index]
        history = state.action_history[0]
        all_actions = state.possible_actions[0]
        action_sequence = [self._get_action_string(all_actions[action]) for action in history]
        return self._check_denotation(action_sequence, instance_label_strings, worlds)

    @staticmethod
    def _get_action_string(production_rule: ProductionRuleArray) -> str:
        return "%s -> %s" % (production_rule["left"][0], production_rule["right"][0])
