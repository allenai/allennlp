from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Linear

from allennlp.common import Registrable
from allennlp.modules.seq2seq_decoders.seq_decoder import SeqDecoder
from allennlp.data import Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_decoders.decoder_cell import DecoderCell
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric


@SeqDecoder.register("rnn_seq_decoder")
class RnnSeqDecoder(SeqDecoder):
    """

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    decoder_cell : ``DecoderCell``, required
        Module that contains implementation of neural network for decoding output elements
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
    scheduled_sampling_ratio : ``float``
        Defines ratio between teacher forced training and real output usage.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            decoder_cell: DecoderCell,
            max_decoding_steps: int,
            bidirectional_input: bool = False,
            beam_size: int = None,
            target_namespace: str = "tokens",
            tensor_based_metric: Metric = None,
            token_based_metric: Metric = None,
            scheduled_sampling_ratio: float = 0.,
    ):

        super(RnnSeqDecoder, self).__init__(
            vocab=vocab,
            target_namespace=target_namespace,
            tensor_based_metric=tensor_based_metric,
            token_based_metric=token_based_metric
        )

        self.bidirectional_input = bidirectional_input

        self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Decodes the sequence of encoded hidden states into e new sequence of hidden states.
        self.decoder_cell = decoder_cell

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.decoder_output_dim = self.decoder_cell.get_output_dim()
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
        self._target_embedder = Embedding(target_vocab_size, self.decoder_cell.target_embedding_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self.decoder_cell.get_output_dim(), target_vocab_size)

        self._scheduled_sampling_ratio = scheduled_sampling_ratio

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step)

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _forward_loss(self, state: Dict[str, torch.Tensor], target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[
        str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1

            # Prepare embeddings for targets. They will be used as gold embeddings during decoder training
            # shape: (batch_size, max_target_sequence_length, embedding_dim)
            target_embeddings = self._target_embedder(targets)
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        # shape: (steps, batch_size, target_embedding_dim)
        steps_embeddings = torch.tensor([])

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size, steps, target_embedding_dim)
                state['previous_steps_predictions'] = steps_embeddings

                # shape: (batch_size, )
                effective_last_prediction = last_predictions
            elif not target_tokens:
                # shape: (batch_size, steps, target_embedding_dim)
                state['previous_steps_predictions'] = steps_embeddings

                # shape: (batch_size, )
                effective_last_prediction = last_predictions
            else:
                # shape: (batch_size, )
                effective_last_prediction = targets[:, timestep]

                if timestep == 0:
                    state['previous_steps_predictions'] = torch.tensor([])
                else:
                    # shape: (batch_size, steps, target_embedding_dim)
                    state['previous_steps_predictions'] = target_embeddings[:, :timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(effective_last_prediction, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            # shape: (batch_size, 1, target_embedding_dim)
            last_predictions_embeddings = self._target_embedder(last_predictions).unsqueeze(1)

            # This step is required, since we want to keep up two different prediction history: gold and real
            if steps_embeddings.shape[-1] == 0:
                # There is no previous steps, except for start vectors in ``last_predictions``
                # shape: (group_size, 1, target_embedding_dim)
                steps_embeddings = last_predictions_embeddings
            else:
                # shape: (group_size, steps_count, target_embedding_dim)
                steps_embeddings = torch.cat([steps_embeddings, last_predictions_embeddings], 1)

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, steps_count, decoder_output_dim)
        previous_steps_predictions = state.get("previous_steps_predictions")

        # shape: (batch_size, 1, target_embedding_dim)
        last_predictions_embeddings = self._target_embedder(last_predictions).unsqueeze(1)

        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in ``last_predictions``
            # shape: (group_size, 1, target_embedding_dim)
            previous_steps_predictions = last_predictions_embeddings
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat([previous_steps_predictions, last_predictions_embeddings], 1)

        decoder_state, decoder_output = self.decoder_cell(
            previous_steps_predictions=previous_steps_predictions,
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
            previous_state=state
        )

        state["previous_steps_predictions"] = previous_steps_predictions

        # Update state with new decoder state, override previous state
        state.update(decoder_state)

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_output)

        return output_projections, state

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
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

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

        state.update(self.decoder_cell.init_decoder_state(batch_size, final_encoder_output))

        return state

    def get_output_dim(self):
        return self.decoder_cell.get_output_dim()

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    def forward(self,
                encoder_out: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None):

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
