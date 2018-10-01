from typing import Dict, List, Tuple

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch


@Model.register("simple_seq2seq")
class SimpleSeq2Seq(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    attention : ``Attention``, required
        The attention function to apply over inputs.
    source_namespace : ``str``, optional (default = 'source_tokens')
    target_namespace : ``str``, optional (default = 'target_tokens')
    target_embedding_dim : ``int``, optional (default = 30)
        You can specify an embedding dimensionality for the target side.
    max_decoding_steps : ``int``, optional (default = 50)
        Maximum length of decoded sequences.
    beam_size : ``int``, optional (default = 5)
        Width of the beam for beam search.
    scheduled_sampling_ratio : ``float``, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 max_decoding_steps: int = 50,
                 beam_size: int = 5,
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 target_embedding_dim: int = 30,
                 scheduled_sampling_ratio: float = 0.) -> None:
        super(SimpleSeq2Seq, self).__init__(vocab)
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Attention mechanism applied to the encoder output for each step.
        self._attention = attention

        # Dense embedding of vocab words in the target space.
        self._target_embedder = Embedding(num_classes, target_embedding_dim)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_output_dim = self.encoder_output_dim

        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.decoder_input_dim = self.decoder_output_dim

        # Our decoder input will be the concatenation of the decoder hidden state and the previous
        # action embedding, and we'll project that down to the decoder's input dimension.
        self._input_projection_layer = \
            Linear(self.encoder_output_dim + target_embedding_dim, self.decoder_input_dim)

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self._decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self.decoder_output_dim, num_classes)

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

    @overrides
    def forward(self,
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        ``Dict[str, torch.Tensor]``
        """
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)

        batch_size, _, _ = embedded_input.size()

        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length)

        encoder_outputs = self._encoder(embedded_input, source_mask)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)

        final_encoder_output = util.get_final_encoder_states(
                encoder_outputs,
                source_mask,
                self._encoder.is_bidirectional())
        # shape: (batch_size, encoder_output_dim)

        # Initialize the decoder hidden state the final output of the encoder.
        decoder_hidden = final_encoder_output
        # shape: (batch_size, decoder_output_dim)

        decoder_context = encoder_outputs.new_zeros(batch_size, self.decoder_output_dim)
        # shape: (batch_size, decoder_output_dim)

        state = {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
                "decoder_hidden": decoder_hidden,
                "decoder_context": decoder_context
        }

        if target_tokens:
            return self._forward_train(target_tokens, state)

        return self._forward_predict(state)

    def _forward_train(self,
                       target_tokens: Dict[str, torch.LongTensor],
                       state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during training."""
        encoder_outputs = state["encoder_outputs"]
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)

        source_mask = state["source_mask"]
        # shape: (batch_size, max_input_sequence_length)

        decoder_hidden = state["decoder_hidden"]
        # shape: (batch_size, decoder_output_dim)

        decoder_context = state["decoder_context"]
        # shape: (batch_size, decoder_output_dim)

        targets = target_tokens["tokens"]
        # shape: (batch_size, max_target_sequence_length)

        batch_size, target_sequence_length = targets.size()

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1

        last_predictions = None
        step_logits: List[torch.Tensor] = []
        step_probabilities: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            # Use gold tokens at a rate of `1 - self._scheduled_sampling_ratio`.
            if torch.rand(1).item() >= self._scheduled_sampling_ratio:
                input_choices = targets[:, timestep]
                # shape: (batch_size,)
            else:
                if timestep == 0:
                    # For the first timestep, when we do not have targets, we input start symbols.
                    input_choices = source_mask.new_full((batch_size,), fill_value=self._start_index)
                    # shape: (batch_size,)
                else:
                    input_choices = last_predictions
                    # shape: (batch_size,)

            embedded_input = self._target_embedder(input_choices)
            # shape: (batch_size, target_embedding_dim)

            attended_input = self._prepare_attended_input(
                    decoder_hidden,
                    encoder_outputs,
                    source_mask)
            # shape: (batch_size, encoder_output_dim)

            decoder_input = self._input_projection_layer(torch.cat((attended_input, embedded_input), -1))
            # shape: (batch_size, decoder_input_dim)

            decoder_hidden, decoder_context = self._decoder_cell(
                    decoder_input,
                    (decoder_hidden, decoder_context))
            # shape (decoder_hidden): (batch_size, decoder_output_dim)
            # shape (decoder_context): (batch_size, decoder_output_dim)

            output_projections = self._output_projection_layer(decoder_hidden)
            # shape: (batch_size, num_classes)

            step_logits.append(output_projections.unsqueeze(1))
            # list of tensors, shape: (batch_size, 1, num_classes)

            class_probabilities = F.softmax(output_projections, dim=-1)
            # shape: (batch_size, num_classes)

            step_probabilities.append(class_probabilities.unsqueeze(1))
            # list of tensors, shape: (batch_size, 1, num_classes)

            _, predicted_classes = torch.max(class_probabilities, 1)
            # shape (predicted_classes): (batch_size,)

            last_predictions = predicted_classes
            # shape (predicted_classes): (batch_size,)

            step_predictions.append(last_predictions.unsqueeze(1))
            # list of tensors, shape: (batch_size, 1)

        logits = torch.cat(step_logits, 1)
        # shape: (batch_size, num_decoding_steps, num_classes)

        class_probabilities = torch.cat(step_probabilities, 1)
        # shape: (batch_size, num_decoding_steps, num_classes)

        all_predictions = torch.cat(step_predictions, 1)
        # shape: (batch_size, num_decoding_steps)

        output_dict = {
                "logits": logits,
                "class_probabilities": class_probabilities,
                "predictions": all_predictions,
        }

        # Compute loss.
        target_mask = util.get_text_field_mask(target_tokens)
        loss = self._get_loss(logits, targets, target_mask)
        output_dict["loss"] = loss

        return output_dict

    def _forward_predict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)

        all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, state, self.take_step)
        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)

        output_dict = {
                "logits": log_probabilities,
                "predictions": all_top_k_predictions,
        }
        return output_dict

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Take a decoding step. This is called by the beam search class."""
        encoder_outputs = state["encoder_outputs"]
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)

        source_mask = state["source_mask"]
        # shape: (group_size, max_input_sequence_length)

        decoder_hidden = state["decoder_hidden"]
        # shape: (group_size, decoder_output_dim)

        decoder_context = state["decoder_context"]
        # shape: (group_size, decoder_output_dim)

        embedded_input = self._target_embedder(last_predictions)
        # shape: (group_size, target_embedding_dim)

        attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)
        # shape: (group_size, encoder_output_dim)

        decoder_input = self._input_projection_layer(torch.cat((attended_input, embedded_input), -1))
        # shape: (group_size, decoder_input_dim)

        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))
        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        output_projections = self._output_projection_layer(decoder_hidden)
        # shape: (group_size, num_classes)

        class_log_probabilities = F.log_softmax(output_projections, dim=-1)
        # shape: (group_size, num_classes)

        return class_log_probabilities, state

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.LongTensor = None,
                                encoder_outputs: torch.LongTensor = None,
                                encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
        """
        Apply attention over encoder outputs and decoder state.

        Parameters
        ----------
        decoder_hidden_state : ``torch.LongTensor``, (batch_size, decoder_output_dim)
            Output of from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : ``torch.LongTensor``, (batch_size, max_input_sequence_length, encoder_output_dim)
            Encoder outputs from all time steps. Needed only if using attention.
        encoder_outputs_mask : ``torch.LongTensor``, (batch_size, max_input_sequence_length, encoder_output_dim)
            Masks on encoder outputs. Needed only if using attention.

        Returns
        -------
        ``torch.Tensor``, (batch_size, encoder_output_dim)
        """
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        encoder_outputs_mask = encoder_outputs_mask.float()
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)

        input_weights = self._attention(
                decoder_hidden_state, encoder_outputs, encoder_outputs_mask)
        # shape: (batch_size, max_input_sequence_length)

        attended_input = util.weighted_sum(encoder_outputs, input_weights)
        # shape: (batch_size, encoder_output_dim)

        return attended_input

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()
        # shape: (batch_size, num_decoding_steps)

        relevant_mask = target_mask[:, 1:].contiguous()
        # shape: (batch_size, num_decoding_steps)

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for top_k in predicted_indices:
            top_k_tokens = []
            for indices in top_k:
                indices = list(indices)
                # Collect indices till the first end_symbol
                if self._end_index in indices:
                    indices = indices[:indices.index(self._end_index)]
                predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                    for x in indices]
                top_k_tokens.append(predicted_tokens)
            all_predicted_tokens.append(top_k_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict
