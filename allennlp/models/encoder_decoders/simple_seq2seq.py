from typing import Dict, List, Set, Mapping, Sequence, Tuple

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
from allennlp.state_machines import BeamSearch, TransitionFunction
from allennlp.state_machines.states import RnnStatelet
from allennlp.state_machines.states import SimpleState


@Model.register("simple_seq2seq")
class SimpleSeq2Seq(TransitionFunction[SimpleState], Model):
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
    beam_search : ``BeamSearch``, required
        Beam search algorithm used for decoding during predicting.
    source_namespace : ``str``, optional (default = 'source_tokens')
    target_namespace : ``str``, optional (default = 'target_tokens')
    target_embedding_dim : ``int``, optional (default = 30)
        You can specify an embedding dimensionality for the target side.
    max_decoding_steps : ``int``, optional (default = 50)
        Maximum length of decoded sequences.
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
                 beam_search: BeamSearch,
                 max_decoding_steps: int = 50,
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 target_embedding_dim: int = 30,
                 scheduled_sampling_ratio: float = 0.) -> None:
        super(SimpleSeq2Seq, self).__init__(vocab)
        self._max_decoding_steps = max_decoding_steps
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._beam_search = beam_search

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        #
        # Input encoding parts.
        #

        self._source_embedder = source_embedder
        self._encoder = encoder
        self.encoder_output_dim = self._encoder.get_output_dim()

        #
        # Output decoding parts.
        #

        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        self._attention = attention
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
        self._decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)
        self._output_projection_layer = Linear(self.decoder_output_dim, num_classes)

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

        if target_tokens:
            return self._forward_train(target_tokens,
                                       source_mask,
                                       decoder_hidden,
                                       decoder_context,
                                       encoder_outputs)

        return self._forward_predict(source_mask,
                                     decoder_hidden,
                                     decoder_context,
                                     encoder_outputs)

    def _forward_train(self,
                       target_tokens: Dict[str, torch.LongTensor],
                       source_mask: torch.Tensor,
                       decoder_hidden: torch.Tensor,
                       decoder_context: torch.Tensor,
                       encoder_outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training.

        Parameters
        ----------
        targets : ``torch.Tensor``, (batch_size, max_target_sequence_length)
        source_mask : ``torch.Tensor``, (batch_size, max_input_sequence_length)
        decoder_hidden : ``torch.Tensor``, (batch_size, decoder_output_dim)
        decoder_context : ``torch.Tensor``, (batch_size, decoder_output_dim)
        encoder_outputs : ``torch.Tensor``, (batch_size, max_input_sequence_length, encoder_output_dim)

        Returns
        -------
        ``Dict[str, torch.Tensor]``
        """
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

    def _forward_predict(self,
                         source_mask: torch.Tensor,
                         decoder_hidden: torch.Tensor,
                         decoder_context: torch.Tensor,
                         encoder_outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during prediction using a beam search.

        Parameters
        ----------
        source_mask : ``torch.Tensor``, (batch_size, max_input_sequence_length)
        decoder_hidden : ``torch.Tensor``, (batch_size, decoder_output_dim)
        decoder_context : ``torch.Tensor``, (batch_size, decoder_output_dim)
        encoder_outputs : ``torch.Tensor``, (batch_size, max_input_sequence_length, encoder_output_dim)

        Returns
        -------
        ``Dict[str, torch.Tensor]``
        """
        initial_state = self._create_initial_state(
                source_mask, decoder_hidden, decoder_context, encoder_outputs)
        best_final_states = self._beam_search.search(self._max_decoding_steps, initial_state, self)
        output_dict = self._gather_final_states(best_final_states)
        return output_dict

    def take_step(self,
                  state: SimpleState,
                  max_actions: int = None,
                  allowed_actions: List[Set[int]] = None) -> List[SimpleState]:
        # pylint: disable=unused-argument
        """
        Take a decoding step. This is called by the beam search class.

        Taking a step in the decoder consists of three main parts.  First, we'll construct the
        input to the decoder and update the decoder's hidden state.  Second, we'll use this new
        hidden state (and maybe other information) to predict an action.  Finally, we will
        construct new states for the next step.  Each new state corresponds to one valid action
        that can be taken from the current state, and they are ordered by their probability of
        being selected.
        """
        group_size = len(state.batch_indices)

        #
        # First, we group all the rnn statelets together so that we can treat them as a batch.
        # Then we find the best `max_actions` predictions for state in the group.
        #

        decoder_hidden = torch.stack([rnn_state.hidden_state for rnn_state in state.rnn_state])
        # shape: (group_size, decoder_output_dim)
        decoder_context = torch.stack([rnn_state.memory_cell for rnn_state in state.rnn_state])
        # shape: (group_size, decoder_output_dim)
        embedded_input = torch.stack([rnn_state.previous_action_embedding for rnn_state in state.rnn_state])
        # shape: (group_size, target_embedding_dim)
        attended_input = torch.stack([rnn_state.attended_input for rnn_state in state.rnn_state])
        # shape: (group_size, encoder_output_dim)
        encoder_outputs = torch.stack([state.rnn_state[0].encoder_outputs[i] for i in state.batch_indices])
        # shape: (group_size, encoder_output_dim)
        source_mask = torch.stack([state.rnn_state[0].encoder_output_mask[i] for i in state.batch_indices])
        # shape: (group_size, encoder_output_dim)

        decoder_input = self._input_projection_layer(torch.cat((attended_input, embedded_input), -1))
        # shape: (group_size, decoder_input_dim)

        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))
        # shape (decoder_hidden): (group_size, decoder_output_dim)
        # shape (decoder_context): (group_size, decoder_output_dim)

        output_projections = self._output_projection_layer(decoder_hidden)
        # shape: (group_size, num_classes)

        class_log_probabilities = F.log_softmax(output_projections, dim=-1)
        # shape: (group_size, num_classes)

        best_log_probs, predicted_classes = torch.topk(class_log_probabilities, max_actions)
        # shape (both): (group_size, max_actions)

        #
        # We now need to construct a new state for each prediction, for each element in the group,
        # which gives us `group_size * max_actions` new states. We then sort the states
        # and return them as a list.
        #

        attended_input = self._prepare_attended_input(
                decoder_hidden,
                encoder_outputs,
                source_mask)
        # shape: (batch_size, encoder_output_dim)

        # Each group index here might get accessed multiple times, and doing the slicing operation
        # each time is more expensive than doing it once upfront.
        decoder_hidden_l: List[torch.Tensor] = [x.squeeze(0) for x in decoder_hidden.chunk(group_size, 0)]
        decoder_context_l: List[torch.Tensor] = [x.squeeze(0) for x in decoder_context.chunk(group_size, 0)]
        attended_input_l: List[torch.Tensor] = [x.squeeze(0) for x in attended_input.chunk(group_size, 0)]

        new_states: List[SimpleState] = []
        for i in range(group_size):
            new_action_embedding = self._target_embedder(predicted_classes[i])
            # shape: (max_actions, target_embedding_dim)
            for j in range(max_actions):
                log_prob = best_log_probs[i][j]
                prediction = predicted_classes[i][j]
                new_batch_indices: List[int] = [state.batch_indices[i]]
                new_action_history: List[List[int]] = [state.action_history[i] + [prediction.item()]]
                new_score: List[torch.Tensor] = [state.score[i] + log_prob]
                rnn_state: List[RnnStatelet] = [RnnStatelet(decoder_hidden_l[i],
                                                            decoder_context_l[i],
                                                            new_action_embedding[j],
                                                            attended_input_l[i],
                                                            state.rnn_state[0].encoder_outputs,
                                                            state.rnn_state[0].encoder_output_mask)]
                new_states.append(SimpleState(new_batch_indices,
                                              new_action_history,
                                              new_score,
                                              rnn_state,
                                              self._end_index))

        new_states.sort(key=lambda x: x.score[0].item(), reverse=True)

        return new_states

    def _print_state(self, state: SimpleState) -> None:
        for i in range(len(state.batch_indices)):
            tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                      for x in state.action_history[i]]
            print(torch.exp(state.score[i]).item(), tokens)

    @staticmethod
    def _gather_final_states(states: Mapping[int, Sequence[SimpleState]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=not-callable
        """
        Take the best state from within each batch and batch together the results.

        This transforms a mapping of batch index -> states to an output_dict that
        has the same format as the output_dict from a training pass.
        """
        batch_results: List[Tuple[int, torch.Tensor, torch.LongTensor]] = []
        for batch_num in states:
            best = states[batch_num][0]
            batch_results.append((batch_num, best.score[0], torch.tensor(best.action_history[0])))
        batch_results.sort(key=lambda x: x[0])
        output_dict = {
                "predictions": torch.stack([x[2] for x in batch_results]),
                "probability": torch.exp(torch.stack([x[1] for x in batch_results])),
        }
        return output_dict

    def _create_initial_state(self,
                              source_mask: torch.Tensor,
                              decoder_hidden: torch.Tensor,
                              decoder_context: torch.Tensor,
                              encoder_outputs: torch.Tensor) -> SimpleState:
        """
        Create initialize state for decoding from start token.

        Parameters
        ----------
        source_mask : ``torch.Tensor``, (batch_size, max_input_sequence_length)
        decoder_hidden : ``torch.Tensor``, (batch_size, decoder_output_dim)
        decoder_context : ``torch.Tensor``, (batch_size, decoder_output_dim)
        encoder_outputs : ``torch.Tensor``, (batch_size, max_input_sequence_length, encoder_output_dim)

        Returns
        -------
        ``SimpleState``
        """
        batch_size = source_mask.size()[0]

        # For the first timestep, when we do not have targets, we input start symbols.
        initial_choices = source_mask.new_full((batch_size,), fill_value=self._start_index)

        initial_embedding = self._target_embedder(initial_choices)
        # shape: (batch_size, target_embedding_dim)

        attended_input = self._prepare_attended_input(
                decoder_hidden,
                encoder_outputs,
                source_mask)
        # shape: (batch_size, encoder_output_dim)

        initial_score = decoder_hidden.new_zeros(batch_size)
        # shape: (batch_size,)

        # `State` classes required these to be in a list:
        encoder_output_list: List[torch.Tensor] = [encoder_outputs[i] for i in range(batch_size)]
        source_mask_list: List[torch.Tensor] = [source_mask[i] for i in range(batch_size)]
        initial_score_list = [initial_score[i] for i in range(batch_size)]

        # Gather rnn statelets.
        initial_rnn_states: List[RnnStatelet] = []
        for i in range(batch_size):
            initial_rnn_states.append(RnnStatelet(decoder_hidden[i],
                                                  decoder_context[i],
                                                  initial_embedding[i],
                                                  attended_input[i],
                                                  encoder_output_list,
                                                  source_mask_list))

        return SimpleState(list(range(batch_size)),
                           [[] for _ in range(batch_size)],
                           initial_score_list,
                           initial_rnn_states,
                           self._end_index)

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
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict
