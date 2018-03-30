import logging
from typing import List, Dict

from overrides import overrides

import torch

from allennlp.common import Params
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.decoding import BeamSearch, MaximumMarginalLikelihood
from allennlp.nn import util as nn_util
from allennlp.models.model import Model
from allennlp.models.encoder_decoders.nlvr.nlvr_decoder_state import NlvrDecoderState
from allennlp.models.encoder_decoders.nlvr.nlvr_decoder_step import NlvrDecoderStep
from allennlp.models.encoder_decoders.nlvr.nlvr_semantic_parser import NlvrSemanticParser
from allennlp.semparse.worlds import NlvrWorld

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("nlvr_direct_parser")
class NlvrDirectSemanticParser(NlvrSemanticParser):
    """
    ``NlvrDirectSemanticParser`` is an ``NlvrSemanticParser`` that gets around the problem of lack
    of logical form annotations by maximizing the marginal likelihood of an approximate set of target
    sequences that yield the correct denotation. The main difference between this parser and
    ``NlvrCoverageSemanticParser`` is that while this parser takes the output of an offline search
    process as the set of target sequences for training, the latter performs search during training.

    Parameters
    ----------
    vocab : ``Vocabulary``
        Passed to super-class.
    sentence_embedder : ``TextFieldEmbedder``
        Passed to super-class.
    nonterminal_embedder : ``TextFieldEmbedder``
        Passed to super-class.
    terminal_embedder : ``TextFieldEmbedder``
        Passed to super-class.
    encoder : ``Seq2SeqEncoder``
        Passed to super-class.
    attention_function : ``SimilarityFunction``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  This is the similarity function we use for that
        attention.
    decoder_beam_search : ``BeamSearch``
        Beam search used to retrieve best sequences after training.
    max_decoding_steps : ``int``
        Maximum number of steps for beam search after training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 nonterminal_embedder: TextFieldEmbedder,
                 terminal_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention_function: SimilarityFunction,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int) -> None:
        super(NlvrDirectSemanticParser, self).__init__(vocab=vocab,
                                                       sentence_embedder=sentence_embedder,
                                                       nonterminal_embedder=nonterminal_embedder,
                                                       terminal_embedder=terminal_embedder,
                                                       encoder=encoder)
        self._decoder_trainer = MaximumMarginalLikelihood()
        action_embedding_dim = nonterminal_embedder.get_output_dim() * 2

        self._decoder_step = NlvrDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                             action_embedding_dim=action_embedding_dim,
                                             attention_function=attention_function)
        self._decoder_beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        self._action_padding_index = -1

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                worlds: List[List[NlvrWorld]],
                actions: List[List[ProductionRuleArray]],
                target_action_sequences: torch.LongTensor = None,
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing type constrained target sequences that maximize coverage of
        their respective agendas, and minimize a denotation based loss.
        """
        batch_size = len(worlds)
        action_embeddings, action_indices, initial_action_embedding = self._embed_actions(actions)

        initial_rnn_state = self._get_initial_rnn_state(sentence, initial_action_embedding)
        initial_score_list = [nn_util.new_variable_with_data(list(sentence.values())[0],
                                                             torch.Tensor([0.0]))
                              for i in range(batch_size)]
        label_strings = self._get_label_strings(labels)
        # TODO (pradeep): Assuming all worlds give the same set of valid actions.
        initial_grammar_state = [self._create_grammar_state(worlds[i][0], actions[i]) for i in
                                 range(batch_size)]
        worlds_list = [worlds[i] for i in range(batch_size)]

        initial_state = NlvrDecoderState(batch_indices=list(range(batch_size)),
                                         action_history=[[] for _ in range(batch_size)],
                                         score=initial_score_list,
                                         rnn_state=initial_rnn_state,
                                         grammar_state=initial_grammar_state,
                                         action_embeddings=action_embeddings,
                                         action_indices=action_indices,
                                         possible_actions=actions,
                                         worlds=worlds_list,
                                         label_strings=label_strings)

        if target_action_sequences is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            target_action_sequences = target_action_sequences.squeeze(-1)
            target_mask = target_action_sequences != self._action_padding_index
        else:
            target_mask = None
        outputs = self._decoder_trainer.decode(initial_state,
                                               self._decoder_step,
                                               (target_action_sequences, target_mask))
        best_final_states = self._decoder_beam_search.search(self._max_decoding_steps,
                                                             initial_state,
                                                             self._decoder_step,
                                                             keep_final_unfinished_states=False)
        best_action_sequences: Dict[int, List[int]] = {}
        for i in range(batch_size):
            # Decoding may not have terminated with any completed logical forms, if `num_steps`
            # isn't long enough (or if the model is not trained enough and gets into an
            # infinite action loop).
            if i in best_final_states:
                best_action_indices = best_final_states[i][0].action_history[0]
                best_action_sequences[i] = best_action_indices
        self._update_metrics(actions=actions,
                             worlds=worlds,
                             best_action_sequences=best_action_sequences,
                             label_strings=label_strings)
        return outputs

    def _update_metrics(self,
                        actions: List[List[ProductionRuleArray]],
                        worlds: List[List[NlvrWorld]],
                        best_action_sequences: Dict[int, List[int]],
                        label_strings: List[List[str]]) -> None:
        batch_size = len(worlds)
        for i in range(batch_size):
            batch_actions = actions[i]
            batch_best_sequences = best_action_sequences[i] if i in best_action_sequences else []
            sequence_is_correct = [False]
            if batch_best_sequences:
                action_strings = [self._get_action_string(batch_actions[rule_id]) for rule_id in
                                  batch_best_sequences]
                instance_label_strings = label_strings[i]
                instance_worlds = worlds[i]
                sequence_is_correct = self._check_denotation(action_strings,
                                                             instance_label_strings,
                                                             instance_worlds)
            for correct_in_world in sequence_is_correct:
                self._denotation_accuracy(1 if correct_in_world else 0)
            self._consistency(1 if all(sequence_is_correct) else 0)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'denotation_accuracy': self._denotation_accuracy.get_metric(reset),
                'consistency': self._consistency.get_metric(reset)
        }

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'NlvrDirectSemanticParser':
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
        decoder_beam_search = BeamSearch.from_params(params.pop("decoder_beam_search"))
        max_decoding_steps = params.pop_int("max_decoding_steps")
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   sentence_embedder=sentence_embedder,
                   nonterminal_embedder=nonterminal_embedder,
                   terminal_embedder=terminal_embedder,
                   encoder=encoder,
                   attention_function=attention_function,
                   decoder_beam_search=decoder_beam_search,
                   max_decoding_steps=max_decoding_steps)
