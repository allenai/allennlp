from typing import Dict, List, Tuple, Optional
from overrides import overrides

import numpy
import torch
import torch.nn.functional as F
from torch.nn import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.modules.seq2seq_decoders.seq_decoder import SeqDecoder
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_decoders.decoder_net import DecoderNet
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric


@SeqDecoder.register("auto_regressive_seq_decoder")
class AutoRegressiveSeqDecoder(SeqDecoder):
    """
    An autoregressive decoder that can be used for most seq2seq tasks.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    decoder_net : `DecoderNet`, required
        Module that contains implementation of neural network for decoding output elements
    max_decoding_steps : `int`, required
        Maximum length of decoded sequences.
    target_embedder : `Embedding`
        Embedder for target tokens.
    target_namespace : `str`, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    beam_size : `int`, optional (default = 4)
        Width of the beam for beam search.
    tensor_based_metric : `Metric`, optional (default = None)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : `Metric`, optional (default = None)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    scheduled_sampling_ratio : `float` optional (default = 0)
        Defines ratio between teacher forced training and real output usage. If its zero
        (teacher forcing only) and `decoder_net`supports parallel decoding, we get the output
        predictions in a single forward pass of the `decoder_net`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        decoder_net: DecoderNet,
        max_decoding_steps: int,
        target_embedder: Embedding,
        target_namespace: str = "tokens",
        tie_output_embedding: bool = False,
        scheduled_sampling_ratio: float = 0,
        label_smoothing_ratio: Optional[float] = None,
        beam_size: int = 4,
        tensor_based_metric: Metric = None,
        token_based_metric: Metric = None,
    ) -> None:
        super().__init__(target_embedder)

        self._vocab = vocab

        # Decodes the sequence of encoded hidden states into e new sequence of hidden states.
        self._decoder_net = decoder_net
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._label_smoothing_ratio = label_smoothing_ratio

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self._vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self._vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        target_vocab_size = self._vocab.get_vocab_size(self._target_namespace)

        if self.target_embedder.get_output_dim() != self._decoder_net.target_embedding_dim:
            raise ConfigurationError(
                "Target Embedder output_dim doesn't match decoder module's input."
            )

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(
            self._decoder_net.get_output_dim(), target_vocab_size
        )

        if tie_output_embedding:
            if self._output_projection_layer.weight.shape != self.target_embedder.weight.shape:
                raise ConfigurationError(
                    "Can't tie embeddings with output linear layer, due to shape mismatch"
                )
            self._output_projection_layer.weight = self.target_embedder.weight

        # These metrics will be updated during training and validation
        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric

        self._scheduled_sampling_ratio = scheduled_sampling_ratio

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the beam search, does beam search and returns beam search results.
        """
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _forward_loss(
        self, state: Dict[str, torch.Tensor], target_tokens: TextFieldTensors
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (batch_size, max_target_sequence_length)
        targets = util.get_token_ids_from_text_field_tensors(target_tokens)

        # Prepare embeddings for targets. They will be used as gold embeddings during decoder training
        # shape: (batch_size, max_target_sequence_length, embedding_dim)
        target_embedding = self.target_embedder(targets)

        # shape: (batch_size, max_target_batch_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)

        if self._scheduled_sampling_ratio == 0 and self._decoder_net.decodes_parallel:
            _, decoder_output = self._decoder_net(
                previous_state=state,
                previous_steps_predictions=target_embedding[:, :-1, :],
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
                previous_steps_mask=target_mask[:, :-1],
            )

            # shape: (group_size, max_target_sequence_length, num_classes)
            logits = self._output_projection_layer(decoder_output)
        else:
            batch_size = source_mask.size()[0]
            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1

            # Initialize target predictions with the start index.
            # shape: (batch_size,)
            last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

            # shape: (steps, batch_size, target_embedding_dim)
            steps_embeddings = torch.Tensor([])

            step_logits: List[torch.Tensor] = []

            for timestep in range(num_decoding_steps):
                if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                    # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                    # during training.
                    # shape: (batch_size, steps, target_embedding_dim)
                    state["previous_steps_predictions"] = steps_embeddings

                    # shape: (batch_size, )
                    effective_last_prediction = last_predictions
                else:
                    # shape: (batch_size, )
                    effective_last_prediction = targets[:, timestep]

                    if timestep == 0:
                        state["previous_steps_predictions"] = torch.Tensor([])
                    else:
                        # shape: (batch_size, steps, target_embedding_dim)
                        state["previous_steps_predictions"] = target_embedding[:, :timestep]

                # shape: (batch_size, num_classes)
                output_projections, state = self._prepare_output_projections(
                    effective_last_prediction, state
                )

                # list of tensors, shape: (batch_size, 1, num_classes)
                step_logits.append(output_projections.unsqueeze(1))

                # shape (predicted_classes): (batch_size,)
                _, predicted_classes = torch.max(output_projections, 1)

                # shape (predicted_classes): (batch_size,)
                last_predictions = predicted_classes

                # shape: (batch_size, 1, target_embedding_dim)
                last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

                # This step is required, since we want to keep up two different prediction history: gold and real
                if steps_embeddings.shape[-1] == 0:
                    # There is no previous steps, except for start vectors in `last_predictions`
                    # shape: (group_size, 1, target_embedding_dim)
                    steps_embeddings = last_predictions_embeddings
                else:
                    # shape: (group_size, steps_count, target_embedding_dim)
                    steps_embeddings = torch.cat([steps_embeddings, last_predictions_embeddings], 1)

            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

        # Compute loss.
        target_mask = util.get_text_field_mask(target_tokens)
        loss = self._get_loss(logits, targets, target_mask)

        # TODO: We will be using beam search to get predictions for validation, but if beam size in 1
        # we could consider taking the last_predictions here and building step_predictions
        # and use that instead of running beam search again, if performance in validation is taking a hit
        output_dict = {"loss": loss}

        return output_dict

    def _prepare_output_projections(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in `last_predictions`
            # shape: (group_size, 1, target_embedding_dim)
            previous_steps_predictions = last_predictions_embeddings
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat(
                [previous_steps_predictions, last_predictions_embeddings], 1
            )

        decoder_state, decoder_output = self._decoder_net(
            previous_state=state,
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
            previous_steps_predictions=previous_steps_predictions,
        )
        state["previous_steps_predictions"] = previous_steps_predictions

        # Update state with new decoder state, override previous state
        state.update(decoder_state)

        if self._decoder_net.decodes_parallel:
            decoder_output = decoder_output[:, -1, :]

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_output)

        return output_projections, state

    def _get_loss(
        self, logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.LongTensor
    ) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of `targets` is expected to be greater than that of `logits` because the
        decoder does not need to compute the output corresponding to the last timestep of
        `targets`. This method aligns the inputs appropriately to compute the loss.

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

        return util.sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask, label_smoothing=self._label_smoothing_ratio
        )

    def get_output_dim(self):
        return self._decoder_net.get_output_dim()

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        # Parameters

        last_predictions : `torch.Tensor`
            A tensor of shape `(group_size,)`, which gives the indices of the predictions
            during the last time step.
        state : `Dict[str, torch.Tensor]`
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape `(group_size, *)`, where `*` can be any other number
            of dimensions.

        # Returns

        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of `(log_probabilities, updated_state)`, where `log_probabilities`
            is a tensor of shape `(group_size, num_classes)` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while `updated_state` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though `group_size` is not necessarily
            equal to `batch_size`, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(
                    self._tensor_based_metric.get_metric(reset=reset)  # type: ignore
                )
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    @overrides
    def forward(
        self, encoder_out: Dict[str, torch.LongTensor], target_tokens: TextFieldTensors = None,
    ) -> Dict[str, torch.Tensor]:
        state = encoder_out
        decoder_init_state = self._decoder_net.init_decoder_state(state)
        state.update(decoder_init_state)

        if target_tokens:
            output_dict = self._forward_loss(state, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

            if target_tokens:
                targets = util.get_token_ids_from_text_field_tensors(target_tokens)
                if self._tensor_based_metric is not None:
                    # shape: (batch_size, beam_size, max_sequence_length)
                    top_k_predictions = output_dict["predictions"]
                    # shape: (batch_size, max_predicted_sequence_length)
                    best_predictions = top_k_predictions[:, 0, :]

                    self._tensor_based_metric(  # type: ignore
                        best_predictions, targets
                    )

                if self._token_based_metric is not None:
                    output_dict = self.post_process(output_dict)
                    predicted_tokens = output_dict["predicted_tokens"]

                    self._token_based_metric(  # type: ignore
                        predicted_tokens, self.indices_to_tokens(targets[:, 1:]),
                    )

        return output_dict

    @overrides
    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        all_predicted_tokens = self.indices_to_tokens(predicted_indices)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def indices_to_tokens(self, batch_indeces: numpy.ndarray) -> List[List[str]]:

        if not isinstance(batch_indeces, numpy.ndarray):
            batch_indeces = batch_indeces.detach().cpu().numpy()

        all_tokens = []
        for indices in batch_indeces:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            tokens = [
                self._vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            all_tokens.append(tokens)

        return all_tokens
