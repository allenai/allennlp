from typing import Dict

import numpy
from overrides import overrides

import torch
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.attention import LegacyAttention
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum


@Model.register("atis")
class Atis(SimpleSeq2Seq):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 attention_function: SimilarityFunction = None,
                 scheduled_sampling_ratio: float = 0.0,
                 ) -> None:

        # super(Atis, self).__init__(vocab, source_embedder, encoder, max_decoding_steps, target_namespace, target_embedding_dim, attention_function, scheduled_sampling_ratio)
        super(SimpleSeq2Seq, self).__init__(vocab)
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._attention_function = attention_function
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        # num_classes = self.vocab.get_vocab_size(self._target_namespace)
        num_classes = 300 #TODO get the size of all the strings?

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder. Also, if
        # we're using attention with ``DotProductSimilarity``, this is needed.
        self._decoder_output_dim = self._encoder.get_output_dim()
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)
        if self._attention_function:
            self._decoder_attention = LegacyAttention(self._attention_function)
            # The output of attention, a weighted average over encoder outputs, will be
            # concatenated to the input vector of the decoder at each time step.
            self._decoder_input_dim = self._encoder.get_output_dim() + target_embedding_dim
        else:
            self._decoder_input_dim = target_embedding_dim
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)


    @overrides
    def _prepare_decode_step_input(self,
                                   input_indices: torch.LongTensor,
                                   decoder_hidden_state: torch.LongTensor = None,
                                   encoder_outputs: torch.LongTensor = None,
                                   encoder_outputs_mask: torch.LongTensor = None) -> torch.LongTensor:
        """
        Given the input indices for the current timestep of the decoder, and all the encoder
        outputs, compute the input at the current timestep.  Note: This method is agnostic to
        whether the indices are gold indices or the predictions made by the decoder at the last
        timestep. So, this can be used even if we're doing some kind of scheduled sampling.

        If we're not using attention, the output of this method is just an embedding of the input
        indices.  If we are, the output will be a concatentation of the embedding and an attended
        average of the encoder inputs.

        Parameters
        ----------
        input_indices : torch.LongTensor
            Indices of either the gold inputs to the decoder or the predicted labels from the
            previous timestep.
        decoder_hidden_state : torch.LongTensor, optional (not needed if no attention)
            Output of from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor, optional (not needed if no attention)
            Encoder outputs from all time steps. Needed only if using attention.
        encoder_outputs_mask : torch.LongTensor, optional (not needed if no attention)
            Masks on encoder outputs. Needed only if using attention.
        """
        # input_indices : (batch_size,)  since we are processing these one timestep at a time.
        # (batch_size, target_embedding_dim)
        embedded_input = self._target_embedder(input_indices)
        print(input_indices.size())

        if self._attention_function:
            # encoder_outputs : (batch_size, input_sequence_length, encoder_output_dim)
            # Ensuring mask is also a FloatTensor. Or else the multiplication within attention will
            # complain.
            encoder_outputs_mask = encoder_outputs_mask.float()
            # (batch_size, input_sequence_length)
            input_weights = self._decoder_attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)
            # (batch_size, encoder_output_dim)
            attended_input = weighted_sum(encoder_outputs, input_weights)
            # (batch_size, encoder_output_dim + target_embedding_dim)

            print('attended_input', attended_input.size())
            print('embedded_input', embedded_input.size())
            print('embedded_input', embedded_input)
            return torch.cat((attended_input, embedded_input), -1)
        else:
            return embedded_input


    @overrides
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                actions=None,
                target_action_sequence: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # (batch_size, input_sequence_length, encoder_output_dim)
        new_target_action_sequence = {"tokens" : target_action_sequence}
        # super(Atis, self).forward(utterance, new_target_action_sequence)

        print('actions', actions)
        print('actions', len(actions[0]))

        source_tokens = utterance
        target_tokens = new_target_action_sequence

        embedded_input = self._source_embedder(source_tokens)
        batch_size, _, _ = embedded_input.size()
        source_mask = get_text_field_mask(source_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        if target_tokens:
            targets = target_tokens["tokens"].squeeze(2)
            target_sequence_length = targets.size()[1]
            # The last input from the target is either padding or the end symbol. Either way, we
            # don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps
        decoder_hidden = final_encoder_output
        decoder_context = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
        print('decoder output dim', self._decoder_output_dim)
        last_predictions = None
        step_logits = []
        step_probabilities = []
        step_predictions = []
        print('num_decoding_steps', num_decoding_steps)
        for timestep in range(num_decoding_steps):
            print('timestep', timestep)
            print('self.trianing', self.training)
            if self.training and torch.rand(1).item() >= self._scheduled_sampling_ratio:
                input_choices = targets[:, timestep]
            else:
                if timestep == 0:
                    # For the first timestep, when we do not have targets, we input start symbols.
                    # (batch_size,)
                    input_choices = source_mask.new_full((batch_size,), fill_value=self._start_index)
                else:
                    input_choices = last_predictions

            print('input choices', input_choices)
            print('input choices', input_choices.size())
            decoder_input = self._prepare_decode_step_input(input_choices, decoder_hidden,
                                                            encoder_outputs, source_mask)

            print('decoder input', decoder_input)
            print('decoder input', decoder_input.size())
            decoder_hidden, decoder_context = self._decoder_cell(decoder_input,
                                                                 (decoder_hidden, decoder_context))
            # (batch_size, num_classes)
            output_projections = self._output_projection_layer(decoder_hidden)
            print('output_projections', output_projections.size())

            # list of (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))
            class_probabilities = F.softmax(output_projections, dim=-1)
            _, predicted_classes = torch.max(class_probabilities, 1)
            step_probabilities.append(class_probabilities.unsqueeze(1))
            last_predictions = predicted_classes
            # (batch_size, 1)
            step_predictions.append(last_predictions.unsqueeze(1))
        # step_logits is a list containing tensors of shape (batch_size, 1, num_classes)
        # This is (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)
        class_probabilities = torch.cat(step_probabilities, 1)
        all_predictions = torch.cat(step_predictions, 1)
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "predictions": all_predictions}
        if target_tokens:
            target_mask = get_text_field_mask(target_tokens)
            print('logits', logits.size())
            print('targets', targets.size())
            print('targets_mask', target_mask.size())

            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss
            # TODO: Define metrics
        return output_dict
