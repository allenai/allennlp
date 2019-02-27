from typing import Dict, List, Tuple

import torch
from torch.nn import Linear, Module
import torch.nn.functional as F

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_decoders.decoder_cell import DecoderCell
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric, BLEU


class SeqDecoder(Module):
    """
    A ``SeqDecoder`` is a base class for different types of Seq decoding modules

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    bidirectional_input : ``bool``
        If input encoded sequence was produced by bidirectional encoder.
        If True, the first encode step of back direction will be used as initial hidden state.
        If not, the will be used the last step only.
    target_namespace : ``str``, optional (default = 'target_tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : ``int``, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    beam_size : ``int``, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    tensor_based_metric : ``Metric``, optional (default = BLEU)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : ``Metric``, optional (default = None)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.

    """

    def __init__(
            self,
            vocab: Vocabulary,
            decoder_cell: DecoderCell,
            max_decoding_steps: int,
            bidirectional_input: bool,
            beam_size: int = None,
            target_embedding_dim: int = None,
            target_namespace: str = "tokens",
            tensor_based_metric: Metric = None,
            token_based_metric: Metric = None,
    ):

        self.vocab = vocab
        super().__init__()

        self._target_namespace = target_namespace

        self.bidirectional_input = bidirectional_input

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token,
                                                     self._target_namespace)  # pylint: disable=protected-access

        # This metrics will be updated during training and validation
        self._tensor_based_metric = tensor_based_metric or \
                                    BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})
        self._token_based_metric = token_based_metric
        self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Decodes the sequence of encoded hidden states into e new sequence of hidden states.
        self._decoder = decoder_cell

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.decoder_output_dim = self._decoder.get_output_dim()
        self.decoder_input_dim = self.decoder_output_dim
        self.encoder_output_dim = self.decoder_input_dim

        target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        # The decoder input will be a function of the embedding of the previous predicted token,
        # an attended encoder hidden state called the "attentive read", and another
        # weighted sum of the encoder hidden state called the "selective read".
        # While the weights for the attentive read are calculated by an `Attention` module,
        # the weights for the selective read are simply the predicted probabilities
        # corresponding to each token in the source sentence that matches the target
        # token from the previous timestep.

        # Dense embedding of vocab words in the target space.
        self._target_embedder = Embedding(target_vocab_size, target_embedding_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder.get_output_dim(), target_vocab_size)

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        raise NotImplementedError()

    def _forward_loss(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        raise NotImplementedError()

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        batch_size, _ = state["source_mask"].size()

        # Initialize the decoder hidden state with the final output of the encoder,
        # and the decoder context with zeros.
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            state["encoder_outputs"],
            state["source_mask"],
            bidirectional=self.bidirectional_input)

        state.update(self._decoder.init_decoder_state(batch_size, final_encoder_output))

        return state

    def forward(self,
                encoder_out: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:

        state = encoder_out

        if target_tokens:
            state = self._init_decoder_state(encoder_out)
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loss(state, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

            if target_tokens:
                if self._tensor_based_metric is not None:
                    # shape: (batch_size, beam_size, max_sequence_length)
                    top_k_predictions = output_dict["predictions"]
                    # shape: (batch_size, max_predicted_sequence_length)
                    best_predictions = top_k_predictions[:, 0, :]
                    # shape: (batch_size, target_sequence_length)

                    self._tensor_based_metric(best_predictions, target_tokens["tokens"])  # type: ignore

                if self._token_based_metric is not None:
                    output_dict = self.decode(output_dict)
                    predicted_tokens = output_dict['predicted_tokens']

                    self._token_based_metric(predicted_tokens,  # type: ignore
                                             [y.text for y in target_tokens["tokens"][1:-1]])

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(reset=reset))  # type: ignore
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics
