import logging
from typing import Dict, Tuple, List, Any, Union

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric, BLEU
from allennlp.nn.beam_search import BeamSearch


logger = logging.getLogger(__name__)


@Model.register("copynet_seq2seq")
class CopyNetSeq2Seq(Model):
    """
    This is an implementation of [CopyNet](https://arxiv.org/pdf/1603.06393).
    CopyNet is a sequence-to-sequence encoder-decoder model with a copying mechanism
    that can copy tokens from the source sentence into the target sentence instead of
    generating all target tokens only from the target vocabulary.

    It is very similar to a typical seq2seq model used in neural machine translation
    tasks, for example, except that in addition to providing a "generation" score at each timestep
    for the tokens in the target vocabulary, it also provides a "copy" score for each
    token that appears in the source sentence. In other words, you can think of CopyNet
    as a seq2seq model with a dynamic target vocabulary that changes based on the tokens
    in the source sentence, allowing it to predict tokens that are out-of-vocabulary (OOV)
    with respect to the actual target vocab.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies.
    source_embedder : `TextFieldEmbedder`, required
        Embedder for source side sequences
    encoder : `Seq2SeqEncoder`, required
        The encoder of the "encoder/decoder" model
    attention : `Attention`, required
        This is used to get a dynamic summary of encoder outputs at each timestep
        when producing the "generation" scores for the target vocab.
    beam_size : `int`, required
        Beam width to use for beam search prediction.
    max_decoding_steps : `int`, required
        Maximum sequence length of target predictions.
    target_embedding_dim : `int`, optional (default = 30)
        The size of the embeddings for the target vocabulary.
    copy_token : `str`, optional (default = '@COPY@')
        The token used to indicate that a target token was copied from the source.
        If this token is not already in your target vocabulary, it will be added.
    source_namespace : `str`, optional (default = 'source_tokens')
        The namespace for the source vocabulary.
    target_namespace : `str`, optional (default = 'target_tokens')
        The namespace for the target vocabulary.
    tensor_based_metric : `Metric`, optional (default = BLEU)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : `Metric`, optional (default = None)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    initializer : `InitializerApplicator`, optional
        An initialization strategy for the model weights.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        attention: Attention,
        beam_size: int,
        max_decoding_steps: int,
        target_embedding_dim: int = 30,
        copy_token: str = "@COPY@",
        source_namespace: str = "source_tokens",
        target_namespace: str = "target_tokens",
        tensor_based_metric: Metric = None,
        token_based_metric: Metric = None,
        initializer: InitializerApplicator = InitializerApplicator(),
    ) -> None:
        super().__init__(vocab)
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._src_start_index = self.vocab.get_token_index(START_SYMBOL, self._source_namespace)
        self._src_end_index = self.vocab.get_token_index(END_SYMBOL, self._source_namespace)
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token, self._target_namespace)
        self._pad_index = self.vocab.get_token_index(
            self.vocab._padding_token, self._target_namespace
        )
        self._copy_index = self.vocab.add_token_to_namespace(copy_token, self._target_namespace)

        self._tensor_based_metric = tensor_based_metric or BLEU(
            exclude_indices={self._pad_index, self._end_index, self._start_index}
        )
        self._token_based_metric = token_based_metric

        self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        # Encoding modules.
        self._source_embedder = source_embedder
        self._encoder = encoder

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_output_dim = self.encoder_output_dim
        self.decoder_input_dim = self.decoder_output_dim

        target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        # The decoder input will be a function of the embedding of the previous predicted token,
        # an attended encoder hidden state called the "attentive read", and another
        # weighted sum of the encoder hidden state called the "selective read".
        # While the weights for the attentive read are calculated by an `Attention` module,
        # the weights for the selective read are simply the predicted probabilities
        # corresponding to each token in the source sentence that matches the target
        # token from the previous timestep.
        self._target_embedder = Embedding(
            num_embeddings=target_vocab_size, embedding_dim=target_embedding_dim
        )
        self._attention = attention
        self._input_projection_layer = Linear(
            target_embedding_dim + self.encoder_output_dim * 2, self.decoder_input_dim
        )

        # We then run the projected decoder input through an LSTM cell to produce
        # the next hidden state.
        self._decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)

        # We create a "generation" score for each token in the target vocab
        # with a linear projection of the decoder hidden state.
        self._output_generation_layer = Linear(self.decoder_output_dim, target_vocab_size)

        # We create a "copying" score for each source token by applying a non-linearity
        # (tanh) to a linear projection of the encoded hidden state for that token,
        # and then taking the dot product of the result with the decoder hidden state.
        self._output_copying_layer = Linear(self.encoder_output_dim, self.decoder_output_dim)

        # At prediction time, we'll use a beam search to find the best target sequence.
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        source_token_ids: torch.Tensor,
        source_to_target: torch.Tensor,
        metadata: List[Dict[str, Any]],
        target_tokens: TextFieldTensors = None,
        target_token_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Make foward pass with decoder logic for producing the entire target sequence.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()` applied on the source `TextField`. This will be
            passed through a `TextFieldEmbedder` and then through an encoder.
        source_token_ids : `torch.Tensor`, required
            Tensor containing IDs that indicate which source tokens match each other.
            Has shape: `(batch_size, trimmed_source_length)`.
        source_to_target : `torch.Tensor`, required
            Tensor containing vocab index of each source token with respect to the
            target vocab namespace. Shape: `(batch_size, trimmed_source_length)`.
        metadata : `List[Dict[str, Any]]`, required
            Metadata field that contains the original source tokens with key 'source_tokens'
            and any other meta fields. When 'target_tokens' is also passed, the metadata
            should also contain the original target tokens with key 'target_tokens'.
        target_tokens : `TextFieldTensors`, optional (default = None)
            Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
            target tokens are also represented as a `TextField` which must contain a "tokens"
            key that uses single ids.
        target_token_ids : `torch.Tensor`, optional (default = None)
            A tensor of shape `(batch_size, target_sequence_length)` which indicates which
            tokens in the target sequence match tokens in the source sequence.

        # Returns

        Dict[str, torch.Tensor]
        """
        state = self._encode(source_tokens)
        state["source_token_ids"] = source_token_ids
        state["source_to_target"] = source_to_target

        if target_tokens:
            state = self._init_decoder_state(state)
            output_dict = self._forward_loss(target_tokens, target_token_ids, state)
        else:
            output_dict = {}

        output_dict["metadata"] = metadata

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
                    gold_tokens = self._gather_extended_gold_tokens(
                        target_tokens["tokens"]["tokens"], source_token_ids, target_token_ids
                    )
                    self._tensor_based_metric(best_predictions, gold_tokens)  # type: ignore
                if self._token_based_metric is not None:
                    predicted_tokens = self._get_predicted_tokens(
                        output_dict["predictions"], metadata, n_best=1
                    )
                    self._token_based_metric(  # type: ignore
                        predicted_tokens, [x["target_tokens"] for x in metadata]
                    )

        return output_dict

    def _gather_extended_gold_tokens(
        self,
        target_tokens: torch.Tensor,
        source_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> torch.LongTensor:
        """
        Modify the gold target tokens relative to the extended vocabulary.

        For gold targets that are OOV but were copied from the source, the OOV index
        will be changed to the index of the first occurence in the source sentence,
        offset by the size of the target vocabulary.

        # Parameters

        target_tokens : `torch.Tensor`
            Shape: `(batch_size, target_sequence_length)`.
        source_token_ids : `torch.Tensor`
            Shape: `(batch_size, trimmed_source_length)`.
        target_token_ids : `torch.Tensor`
            Shape: `(batch_size, target_sequence_length)`.

        # Returns

        torch.Tensor
            Modified `target_tokens` with OOV indices replaced by offset index
            of first match in source sentence.
        """
        batch_size, target_sequence_length = target_tokens.size()
        trimmed_source_length = source_token_ids.size(1)
        # Only change indices for tokens that were OOV in target vocab but copied from source.
        # shape: (batch_size, target_sequence_length)
        oov = target_tokens == self._oov_index
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        expanded_source_token_ids = source_token_ids.unsqueeze(1).expand(
            batch_size, target_sequence_length, trimmed_source_length
        )
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        expanded_target_token_ids = target_token_ids.unsqueeze(-1).expand(
            batch_size, target_sequence_length, trimmed_source_length
        )
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        matches = expanded_source_token_ids == expanded_target_token_ids
        # shape: (batch_size, target_sequence_length)
        copied = matches.sum(-1) > 0
        # shape: (batch_size, target_sequence_length)
        mask = (oov & copied).long()
        # shape: (batch_size, target_sequence_length)
        first_match = ((matches.cumsum(-1) == 1) * matches).to(torch.uint8).argmax(-1)
        # shape: (batch_size, target_sequence_length)
        new_target_tokens = (
            target_tokens * (1 - mask) + (first_match.long() + self._target_vocab_size) * mask
        )
        return new_target_tokens

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        batch_size, _ = state["source_mask"].size()

        # Initialize the decoder hidden state with the final output of the encoder,
        # and the decoder context with zeros.
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            state["encoder_outputs"], state["source_mask"], self._encoder.is_bidirectional()
        )
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(
            batch_size, self.decoder_output_dim
        )

        return state

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode source input sentences.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _decoder_step(
        self,
        last_predictions: torch.Tensor,
        selective_weights: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = state["source_mask"].float()
        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)
        # shape: (group_size, max_input_sequence_length)
        attentive_weights = self._attention(
            state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask
        )
        # shape: (group_size, encoder_output_dim)
        attentive_read = util.weighted_sum(state["encoder_outputs"], attentive_weights)
        # shape: (group_size, encoder_output_dim)
        selective_read = util.weighted_sum(state["encoder_outputs"][:, 1:-1], selective_weights)
        # shape: (group_size, target_embedding_dim + encoder_output_dim * 2)
        decoder_input = torch.cat((embedded_input, attentive_read, selective_read), -1)
        # shape: (group_size, decoder_input_dim)
        projected_decoder_input = self._input_projection_layer(decoder_input)

        state["decoder_hidden"], state["decoder_context"] = self._decoder_cell(
            projected_decoder_input, (state["decoder_hidden"], state["decoder_context"])
        )
        return state

    def _get_generation_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._output_generation_layer(state["decoder_hidden"])

    def _get_copy_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch_size, max_input_sequence_length - 2, encoder_output_dim)
        trimmed_encoder_outputs = state["encoder_outputs"][:, 1:-1]
        # shape: (batch_size, max_input_sequence_length - 2, decoder_output_dim)
        copy_projection = self._output_copying_layer(trimmed_encoder_outputs)
        # shape: (batch_size, max_input_sequence_length - 2, decoder_output_dim)
        copy_projection = torch.tanh(copy_projection)
        # shape: (batch_size, max_input_sequence_length - 2)
        copy_scores = copy_projection.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1)
        return copy_scores

    def _get_ll_contrib(
        self,
        generation_scores: torch.Tensor,
        generation_scores_mask: torch.Tensor,
        copy_scores: torch.Tensor,
        target_tokens: torch.Tensor,
        target_to_source: torch.Tensor,
        copy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the log-likelihood contribution from a single timestep.

        # Parameters

        generation_scores : `torch.Tensor`
            Shape: `(batch_size, target_vocab_size)`
        generation_scores_mask : `torch.Tensor`
            Shape: `(batch_size, target_vocab_size)`. This is just a tensor of 1's.
        copy_scores : `torch.Tensor`
            Shape: `(batch_size, trimmed_source_length)`
        target_tokens : `torch.Tensor`
            Shape: `(batch_size,)`
        target_to_source : `torch.Tensor`
            Shape: `(batch_size, trimmed_source_length)`
        copy_mask : `torch.Tensor`
            Shape: `(batch_size, trimmed_source_length)`

        # Returns

        Tuple[torch.Tensor, torch.Tensor]
            Shape: `(batch_size,), (batch_size, max_input_sequence_length)`
        """
        _, target_size = generation_scores.size()

        # The point of this mask is to just mask out all source token scores
        # that just represent padding. We apply the mask to the concatenation
        # of the generation scores and the copy scores to normalize the scores
        # correctly during the softmax.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat((generation_scores_mask, copy_mask), dim=-1)
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        log_probs = util.masked_log_softmax(all_scores, mask)
        # Calculate the log probability (`copy_log_probs`) for each token in the source sentence
        # that matches the current target token. We use the sum of these copy probabilities
        # for matching tokens in the source sentence to get the total probability
        # for the target token. We also need to normalize the individual copy probabilities
        # to create `selective_weights`, which are used in the next timestep to create
        # a selective read state.
        # shape: (batch_size, trimmed_source_length)
        copy_log_probs = log_probs[:, target_size:] + (target_to_source.float() + 1e-45).log()
        # Since `log_probs[:, target_size]` gives us the raw copy log probabilities,
        # we use a non-log softmax to get the normalized non-log copy probabilities.
        selective_weights = util.masked_softmax(log_probs[:, target_size:], target_to_source)
        # This mask ensures that item in the batch has a non-zero generation probabilities
        # for this timestep only when the gold target token is not OOV or there are no
        # matching tokens in the source sentence.
        # shape: (batch_size, 1)
        gen_mask = ((target_tokens != self._oov_index) | (target_to_source.sum(-1) == 0)).float()
        log_gen_mask = (gen_mask + 1e-45).log().unsqueeze(-1)
        # Now we get the generation score for the gold target token.
        # shape: (batch_size, 1)
        generation_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)) + log_gen_mask
        # ... and add the copy score to get the step log likelihood.
        # shape: (batch_size, 1 + trimmed_source_length)
        combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
        # shape: (batch_size,)
        step_log_likelihood = util.logsumexp(combined_gen_and_copy)

        return step_log_likelihood, selective_weights

    def _forward_loss(
        self,
        target_tokens: TextFieldTensors,
        target_token_ids: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss against gold targets.
        """
        batch_size, target_sequence_length = target_tokens["tokens"]["tokens"].size()

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1
        # We use this to fill in the copy index when the previous input was copied.
        # shape: (batch_size,)
        copy_input_choices = source_mask.new_full((batch_size,), fill_value=self._copy_index)
        # shape: (batch_size, trimmed_source_length)
        copy_mask = source_mask[:, 1:-1].float()
        # We need to keep track of the probabilities assigned to tokens in the source
        # sentence that were copied during the previous timestep, since we use
        # those probabilities as weights when calculating the "selective read".
        # shape: (batch_size, trimmed_source_length)
        selective_weights = state["decoder_hidden"].new_zeros(copy_mask.size())

        # Indicates which tokens in the source sentence match the current target token.
        # shape: (batch_size, trimmed_source_length)
        target_to_source = state["source_token_ids"].new_zeros(copy_mask.size())

        # This is just a tensor of ones which we use repeatedly in `self._get_ll_contrib`,
        # so we create it once here to avoid doing it over-and-over.
        generation_scores_mask = state["decoder_hidden"].new_full(
            (batch_size, self._target_vocab_size), fill_value=1.0
        )

        step_log_likelihoods = []
        for timestep in range(num_decoding_steps):
            # shape: (batch_size,)
            input_choices = target_tokens["tokens"]["tokens"][:, timestep]
            # If the previous target token was copied, we use the special copy token.
            # But the end target token will always be THE end token, so we know
            # it was not copied.
            if timestep < num_decoding_steps - 1:
                # Get mask tensor indicating which instances were copied.
                # shape: (batch_size,)
                copied = (
                    (input_choices == self._oov_index) & (target_to_source.sum(-1) > 0)
                ).long()
                # shape: (batch_size,)
                input_choices = input_choices * (1 - copied) + copy_input_choices * copied
                # shape: (batch_size, trimmed_source_length)
                target_to_source = state["source_token_ids"] == target_token_ids[
                    :, timestep + 1
                ].unsqueeze(-1)
            # Update the decoder state by taking a step through the RNN.
            state = self._decoder_step(input_choices, selective_weights, state)
            # Get generation scores for each token in the target vocab.
            # shape: (batch_size, target_vocab_size)
            generation_scores = self._get_generation_scores(state)
            # Get copy scores for each token in the source sentence, excluding the start
            # and end tokens.
            # shape: (batch_size, trimmed_source_length)
            copy_scores = self._get_copy_scores(state)
            # shape: (batch_size,)
            step_target_tokens = target_tokens["tokens"]["tokens"][:, timestep + 1]
            step_log_likelihood, selective_weights = self._get_ll_contrib(
                generation_scores,
                generation_scores_mask,
                copy_scores,
                step_target_tokens,
                target_to_source,
                copy_mask,
            )
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

        # Gather step log-likelihoods.
        # shape: (batch_size, num_decoding_steps = target_sequence_length - 1)
        log_likelihoods = torch.cat(step_log_likelihoods, 1)
        # Get target mask to exclude likelihood contributions from timesteps after
        # the END token.
        # shape: (batch_size, target_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)
        # The first timestep is just the START token, which is not included in the likelihoods.
        # shape: (batch_size, num_decoding_steps)
        target_mask = target_mask[:, 1:].float()
        # Sum of step log-likelihoods.
        # shape: (batch_size,)
        log_likelihood = (log_likelihoods * target_mask).sum(dim=-1)
        # The loss is the negative log-likelihood, averaged over the batch.
        loss = -log_likelihood.sum() / batch_size

        return {"loss": loss}

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, source_length = state["source_mask"].size()
        trimmed_source_length = source_length - 2
        # Initialize the copy scores to zero.
        state["copy_log_probs"] = (
            state["decoder_hidden"].new_zeros((batch_size, trimmed_source_length)) + 1e-45
        ).log()
        # shape: (batch_size,)
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index
        )
        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_search_step
        )
        return {"predicted_log_probs": log_probabilities, "predictions": all_top_k_predictions}

    def _get_input_and_selective_weights(
        self, last_predictions: torch.LongTensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Get input choices for the decoder and the selective copy weights.

        The decoder input choices are simply the `last_predictions`, except for
        target OOV predictions that were copied from source tokens, in which case
        the prediction will be changed to the COPY symbol in the target namespace.

        The selective weights are just the probabilities assigned to source
        tokens that were copied, normalized to sum to 1. If no source tokens were copied,
        there will be all zeros.

        # Parameters

        last_predictions : `torch.LongTensor`
            Shape: `(group_size,)`
        state : `Dict[str, torch.Tensor]`

        # Returns

        Tuple[torch.LongTensor, torch.Tensor]
            `input_choices` (shape `(group_size,)`) and `selective_weights`
            (shape `(group_size, trimmed_source_length)`).
        """
        group_size, trimmed_source_length = state["source_to_target"].size()

        # This is a mask indicating which last predictions were copied from the
        # the source AND not in the target vocabulary (OOV).
        # (group_size,)
        only_copied_mask = (last_predictions >= self._target_vocab_size).long()

        # If the last prediction was in the target vocab or OOV but not copied,
        # we use that as input, otherwise we use the COPY token.
        # shape: (group_size,)
        copy_input_choices = only_copied_mask.new_full((group_size,), fill_value=self._copy_index)
        input_choices = (
            last_predictions * (1 - only_copied_mask) + copy_input_choices * only_copied_mask
        )

        # In order to get the `selective_weights`, we need to find out which predictions
        # were copied or copied AND generated, which is the case when a prediction appears
        # in both the source sentence and the target vocab. But whenever a prediction
        # is in the target vocab (even if it also appeared in the source sentence),
        # its index will be the corresponding target vocab index, not its index in
        # the source sentence offset by the target vocab size. So we first
        # use `state["source_to_target"]` to get an indicator of every source token
        # that matches the predicted target token.
        # shape: (group_size, trimmed_source_length)
        expanded_last_predictions = last_predictions.unsqueeze(-1).expand(
            group_size, trimmed_source_length
        )
        # shape: (group_size, trimmed_source_length)
        source_copied_and_generated = (
            state["source_to_target"] == expanded_last_predictions
        ).long()

        # In order to get indicators for copied source tokens that are OOV with respect
        # to the target vocab, we'll make use of `state["source_token_ids"]`.
        # First we adjust predictions relative to the start of the source tokens.
        # This makes sense because predictions for copied tokens are given by the index of the copied
        # token in the source sentence, offset by the size of the target vocabulary.
        # shape: (group_size,)
        adjusted_predictions = last_predictions - self._target_vocab_size
        # The adjusted indices for items that were not copied will be negative numbers,
        # and therefore invalid. So we zero them out.
        adjusted_predictions = adjusted_predictions * only_copied_mask
        # shape: (group_size, trimmed_source_length)
        source_token_ids = state["source_token_ids"]
        # shape: (group_size, trimmed_source_length)
        adjusted_prediction_ids = source_token_ids.gather(-1, adjusted_predictions.unsqueeze(-1))
        # This mask will contain indicators for source tokens that were copied
        # during the last timestep.
        # shape: (group_size, trimmed_source_length)
        source_only_copied = (source_token_ids == adjusted_prediction_ids).long()
        # Since we zero'd-out indices for predictions that were not copied,
        # we need to zero out all entries of this mask corresponding to those predictions.
        source_only_copied = source_only_copied * only_copied_mask.unsqueeze(-1)

        # shape: (group_size, trimmed_source_length)
        mask = source_only_copied | source_copied_and_generated
        # shape: (group_size, trimmed_source_length)
        selective_weights = util.masked_softmax(state["copy_log_probs"], mask)

        return input_choices, selective_weights

    def _gather_final_log_probs(
        self,
        generation_log_probs: torch.Tensor,
        copy_log_probs: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Combine copy probabilities with generation probabilities for matching tokens.

        # Parameters

        generation_log_probs : `torch.Tensor`
            Shape: `(group_size, target_vocab_size)`
        copy_log_probs : `torch.Tensor`
            Shape: `(group_size, trimmed_source_length)`
        state : `Dict[str, torch.Tensor]`

        # Returns

        torch.Tensor
            Shape: `(group_size, target_vocab_size + trimmed_source_length)`.
        """
        _, trimmed_source_length = state["source_to_target"].size()
        source_token_ids = state["source_token_ids"]

        # shape: [(batch_size, *)]
        modified_log_probs_list: List[torch.Tensor] = []
        for i in range(trimmed_source_length):
            # shape: (group_size,)
            copy_log_probs_slice = copy_log_probs[:, i]
            # `source_to_target` is a matrix of shape (group_size, trimmed_source_length)
            # where element (i, j) is the vocab index of the target token that matches the jth
            # source token in the ith group, if there is one, or the index of the OOV symbol otherwise.
            # We'll use this to add copy scores to corresponding generation scores.
            # shape: (group_size,)
            source_to_target_slice = state["source_to_target"][:, i]
            # The OOV index in the source_to_target_slice indicates that the source
            # token is not in the target vocab, so we don't want to add that copy score
            # to the OOV token.
            copy_log_probs_to_add_mask = (source_to_target_slice != self._oov_index).float()
            copy_log_probs_to_add = (
                copy_log_probs_slice + (copy_log_probs_to_add_mask + 1e-45).log()
            )
            # shape: (batch_size, 1)
            copy_log_probs_to_add = copy_log_probs_to_add.unsqueeze(-1)
            # shape: (batch_size, 1)
            selected_generation_log_probs = generation_log_probs.gather(
                1, source_to_target_slice.unsqueeze(-1)
            )
            combined_scores = util.logsumexp(
                torch.cat((selected_generation_log_probs, copy_log_probs_to_add), dim=1)
            )
            generation_log_probs = generation_log_probs.scatter(
                -1, source_to_target_slice.unsqueeze(-1), combined_scores.unsqueeze(-1)
            )
            # We have to combine copy scores for duplicate source tokens so that
            # we can find the overall most likely source token. So, if this is the first
            # occurence of this particular source token, we add the log_probs from all other
            # occurences, otherwise we zero it out since it was already accounted for.
            if i < (trimmed_source_length - 1):
                # Sum copy scores from future occurences of source token.
                # shape: (group_size, trimmed_source_length - i)
                source_future_occurences = (
                    source_token_ids[:, (i + 1) :] == source_token_ids[:, i].unsqueeze(-1)
                ).float()  # noqa
                # shape: (group_size, trimmed_source_length - i)
                future_copy_log_probs = (
                    copy_log_probs[:, (i + 1) :] + (source_future_occurences + 1e-45).log()
                )
                # shape: (group_size, 1 + trimmed_source_length - i)
                combined = torch.cat(
                    (copy_log_probs_slice.unsqueeze(-1), future_copy_log_probs), dim=-1
                )
                # shape: (group_size,)
                copy_log_probs_slice = util.logsumexp(combined)
            if i > 0:
                # Remove copy log_probs that we have already accounted for.
                # shape: (group_size, i)
                source_previous_occurences = source_token_ids[:, 0:i] == source_token_ids[
                    :, i
                ].unsqueeze(-1)
                # shape: (group_size,)
                duplicate_mask = (source_previous_occurences.sum(dim=-1) == 0).float()
                copy_log_probs_slice = copy_log_probs_slice + (duplicate_mask + 1e-45).log()

            # Finally, we zero-out copy scores that we added to the generation scores
            # above so that we don't double-count them.
            # shape: (group_size,)
            left_over_copy_log_probs = (
                copy_log_probs_slice + (1.0 - copy_log_probs_to_add_mask + 1e-45).log()
            )
            modified_log_probs_list.append(left_over_copy_log_probs.unsqueeze(-1))
        modified_log_probs_list.insert(0, generation_log_probs)

        # shape: (group_size, target_vocab_size + trimmed_source_length)
        modified_log_probs = torch.cat(modified_log_probs_list, dim=-1)

        return modified_log_probs

    def take_search_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        This function is what gets passed to the `BeamSearch.search` method. It takes
        predictions from the last timestep and the current state and outputs
        the log probabilities assigned to tokens for the next timestep, as well as the updated
        state.

        Since we are predicting tokens out of the extended vocab (target vocab + all unique
        tokens from the source sentence), this is a little more complicated that just
        making a forward pass through the model. The output log probs will have
        shape `(group_size, target_vocab_size + trimmed_source_length)` so that each
        token in the target vocab and source sentence are assigned a probability.

        Note that copy scores are assigned to each source token based on their position, not unique value.
        So if a token appears more than once in the source sentence, it will have more than one score.
        Further, if a source token is also part of the target vocab, its final score
        will be the sum of the generation and copy scores. Therefore, in order to
        get the score for all tokens in the extended vocab at this step,
        we have to combine copy scores for re-occuring source tokens and potentially
        add them to the generation scores for the matching token in the target vocab, if
        there is one.

        So we can break down the final log probs output as the concatenation of two
        matrices, A: `(group_size, target_vocab_size)`, and B: `(group_size, trimmed_source_length)`.
        Matrix A contains the sum of the generation score and copy scores (possibly 0)
        for each target token. Matrix B contains left-over copy scores for source tokens
        that do NOT appear in the target vocab, with zeros everywhere else. But since
        a source token may appear more than once in the source sentence, we also have to
        sum the scores for each appearance of each unique source token. So matrix B
        actually only has non-zero values at the first occurence of each source token
        that is not in the target vocab.

        # Parameters

        last_predictions : `torch.Tensor`
            Shape: `(group_size,)`

        state : `Dict[str, torch.Tensor]`
            Contains all state tensors necessary to produce generation and copy scores
            for next step.

        Notes
        -----
        `group_size` != `batch_size`. In fact, `group_size` = `batch_size * beam_size`.
        """
        _, trimmed_source_length = state["source_to_target"].size()

        # Get input to the decoder RNN and the selective weights. `input_choices`
        # is the result of replacing target OOV tokens in `last_predictions` with the
        # copy symbol. `selective_weights` consist of the normalized copy probabilities
        # assigned to the source tokens that were copied. If no tokens were copied,
        # there will be all zeros.
        # shape: (group_size,), (group_size, trimmed_source_length)
        input_choices, selective_weights = self._get_input_and_selective_weights(
            last_predictions, state
        )
        # Update the decoder state by taking a step through the RNN.
        state = self._decoder_step(input_choices, selective_weights, state)
        # Get the un-normalized generation scores for each token in the target vocab.
        # shape: (group_size, target_vocab_size)
        generation_scores = self._get_generation_scores(state)
        # Get the un-normalized copy scores for each token in the source sentence,
        # excluding the start and end tokens.
        # shape: (group_size, trimmed_source_length)
        copy_scores = self._get_copy_scores(state)
        # Concat un-normalized generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # shape: (group_size, trimmed_source_length)
        copy_mask = state["source_mask"][:, 1:-1].float()
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat(
            (generation_scores.new_full(generation_scores.size(), 1.0), copy_mask), dim=-1
        )
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        log_probs = util.masked_log_softmax(all_scores, mask)
        # shape: (group_size, target_vocab_size), (group_size, trimmed_source_length)
        generation_log_probs, copy_log_probs = log_probs.split(
            [self._target_vocab_size, trimmed_source_length], dim=-1
        )
        # Update copy_probs needed for getting the `selective_weights` at the next timestep.
        state["copy_log_probs"] = copy_log_probs
        # We now have normalized generation and copy scores, but to produce the final
        # score for each token in the extended vocab, we have to go through and add
        # the copy scores to the generation scores of matching target tokens, and sum
        # the copy scores of duplicate source tokens.
        # shape: (group_size, target_vocab_size + trimmed_source_length)
        final_log_probs = self._gather_final_log_probs(generation_log_probs, copy_log_probs, state)

        return final_log_probs, state

    def _get_predicted_tokens(
        self,
        predicted_indices: Union[torch.Tensor, numpy.ndarray],
        batch_metadata: List[Any],
        n_best: int = None,
    ) -> List[Union[List[List[str]], List[str]]]:
        """
        Convert predicted indices into tokens.

        If `n_best = 1`, the result type will be `List[List[str]]`. Otherwise the result
        type will be `List[List[List[str]]]`.
        """
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        predicted_tokens: List[Union[List[List[str]], List[str]]] = []
        for top_k_predictions, metadata in zip(predicted_indices, batch_metadata):
            batch_predicted_tokens: List[List[str]] = []
            for indices in top_k_predictions[:n_best]:
                tokens: List[str] = []
                indices = list(indices)
                if self._end_index in indices:
                    indices = indices[: indices.index(self._end_index)]
                for index in indices:
                    if index >= self._target_vocab_size:
                        adjusted_index = index - self._target_vocab_size
                        token = metadata["source_tokens"][adjusted_index]
                    else:
                        token = self.vocab.get_token_from_index(index, self._target_namespace)
                    tokens.append(token)
                batch_predicted_tokens.append(tokens)
            if n_best == 1:
                predicted_tokens.append(batch_predicted_tokens[0])
            else:
                predicted_tokens.append(batch_predicted_tokens)
        return predicted_tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Finalize predictions.

        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices.
        """
        predicted_tokens = self._get_predicted_tokens(
            output_dict["predictions"], output_dict["metadata"]
        )
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict

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
