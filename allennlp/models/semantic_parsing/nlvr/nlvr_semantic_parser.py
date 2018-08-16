import logging
from typing import List, Dict

from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.models.semantic_parsing.wikitables.grammar_based_decoder_state import GrammarBasedDecoderState
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.nn import util
from allennlp.nn.decoding import GrammarState, RnnState
from allennlp.semparse.type_declarations import type_declaration
from allennlp.semparse.type_declarations.type_declaration import START_SYMBOL
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
    action_embedding_dim : ``int``
        Dimension to use for action embeddings.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input question.
    dropout : ``float``, optional (default=0.0)
        Dropout on the encoder outputs.
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels') -> None:
        super(NlvrSemanticParser, self).__init__(vocab=vocab)

        self._sentence_embedder = sentence_embedder
        self._denotation_accuracy = Average()
        self._consistency = Average()
        self._encoder = encoder
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace

        self._action_embedder = Embedding(num_embeddings=vocab.get_vocab_size(self._rule_namespace),
                                          embedding_dim=action_embedding_dim)

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)

    @overrides
    def forward(self):  # type: ignore
        # pylint: disable=arguments-differ
        # Sub-classes should define their own logic here.
        raise NotImplementedError

    def _get_initial_rnn_state(self, sentence: Dict[str, torch.LongTensor]):
        embedded_input = self._sentence_embedder(sentence)
        # (batch_size, sentence_length)
        sentence_mask = util.get_text_field_mask(sentence).float()

        batch_size = embedded_input.size(0)

        # (batch_size, sentence_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(embedded_input, sentence_mask))

        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             sentence_mask,
                                                             self._encoder.is_bidirectional())
        memory_cell = encoder_outputs.new_zeros(batch_size, self._encoder.get_output_dim())
        attended_sentence, _ = self._decoder_step.attend_on_question(final_encoder_output,
                                                                     encoder_outputs, sentence_mask)
        encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
        sentence_mask_list = [sentence_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnState(final_encoder_output[i],
                                              memory_cell[i],
                                              self._first_action_embedding,
                                              attended_sentence[i],
                                              encoder_outputs_list,
                                              sentence_mask_list))
        return initial_rnn_state

    def _get_label_strings(self, labels):
        # TODO (pradeep): Use an unindexed field for labels?
        labels_data = labels.detach().cpu()
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

    @classmethod
    def _get_action_strings(cls,
                            possible_actions: List[List[ProductionRuleArray]],
                            action_indices: Dict[int, List[List[int]]]) -> List[List[List[str]]]:
        """
        Takes a list of possible actions and indices of decoded actions into those possible actions
        for a batch and returns sequences of action strings. We assume ``action_indices`` is a dict
        mapping batch indices to k-best decoded sequence lists.
        """
        all_action_strings: List[List[List[str]]] = []
        batch_size = len(possible_actions)
        for i in range(batch_size):
            batch_actions = possible_actions[i]
            batch_best_sequences = action_indices[i] if i in action_indices else []
            # This will append an empty list to ``all_action_strings`` if ``batch_best_sequences``
            # is empty.
            action_strings = [[batch_actions[rule_id][0] for rule_id in sequence]
                              for sequence in batch_best_sequences]
            all_action_strings.append(action_strings)
        return all_action_strings

    @staticmethod
    def _get_denotations(action_strings: List[List[List[str]]],
                         worlds: List[List[NlvrWorld]]) -> List[List[List[str]]]:
        all_denotations: List[List[List[str]]] = []
        for instance_worlds, instance_action_sequences in zip(worlds, action_strings):
            denotations: List[List[str]] = []
            for instance_action_strings in instance_action_sequences:
                if not instance_action_strings:
                    continue
                logical_form = instance_worlds[0].get_logical_form(instance_action_strings)
                instance_denotations: List[str] = []
                for world in instance_worlds:
                    # Some of the worlds can be None for instances that come with less than 4 worlds
                    # because of padding.
                    if world is not None:
                        instance_denotations.append(str(world.execute(logical_form)))
                denotations.append(instance_denotations)
            all_denotations.append(denotations)
        return all_denotations

    @staticmethod
    def _check_denotation(action_sequence: List[str],
                          labels: List[str],
                          worlds: List[NlvrWorld]) -> List[bool]:
        is_correct = []
        for world, label in zip(worlds, labels):
            logical_form = world.get_logical_form(action_sequence)
            denotation = world.execute(logical_form)
            is_correct.append(str(denotation).lower() == label)
        return is_correct

    def _create_grammar_state(self,
                              world: NlvrWorld,
                              possible_actions: List[ProductionRuleArray]) -> GrammarState:
        valid_actions = world.get_valid_actions()
        action_mapping = {}
        for i, action in enumerate(possible_actions):
            action_mapping[action[0]] = i
        translated_valid_actions = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.
            action_indices = [action_mapping[action_string] for action_string in action_strings]
            # All actions in NLVR are global actions.
            global_actions = [(possible_actions[index][2], index) for index in action_indices]

            # Then we get the embedded representations of the global actions.
            global_action_tensors, global_action_ids = zip(*global_actions)
            global_action_tensor = torch.cat(global_action_tensors, dim=0)
            global_input_embeddings = self._action_embedder(global_action_tensor)
            translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                       global_input_embeddings,
                                                       list(global_action_ids))
        return GrammarState([START_SYMBOL],
                            {},
                            translated_valid_actions,
                            {},
                            type_declaration.is_nonterminal)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. We only transform the action string sequences into logical
        forms here.
        """
        best_action_strings = output_dict["best_action_strings"]
        # Instantiating an empty world for getting logical forms.
        world = NlvrWorld([])
        logical_forms = []
        for instance_action_sequences in best_action_strings:
            instance_logical_forms = []
            for action_strings in instance_action_sequences:
                if action_strings:
                    instance_logical_forms.append(world.get_logical_form(action_strings))
                else:
                    instance_logical_forms.append('')
            logical_forms.append(instance_logical_forms)
        output_dict["logical_form"] = logical_forms
        return output_dict

    def _check_state_denotations(self, state: GrammarBasedDecoderState, worlds: List[NlvrWorld]) -> List[bool]:
        """
        Returns whether action history in the state evaluates to the correct denotations over all
        worlds. Only defined when the state is finished.
        """
        assert state.is_finished(), "Cannot compute denotations for unfinished states!"
        # Since this is a finished state, its group size must be 1.
        batch_index = state.batch_indices[0]
        instance_label_strings = state.extras[batch_index]
        history = state.action_history[0]
        all_actions = state.possible_actions[0]
        action_sequence = [all_actions[action][0] for action in history]
        return self._check_denotation(action_sequence, instance_label_strings, worlds)
