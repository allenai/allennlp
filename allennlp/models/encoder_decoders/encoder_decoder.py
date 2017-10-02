from typing import Dict

import numpy
from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.encoder_decoder import START_SYMBOL, END_SYMBOL
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum


@Model.register("encoder_decoder")
class EncoderDecoder(Model):
    """
    An ``EncoderDecoder`` is a :class:`Model` which takes a sequence, encodes it, and then uses the encoded
    representations to decode another sequence. It takes an encoder (:class:`Seq2SeqEncoder`) as an
    input. This class implements the functionality of the decoder.

    There are several things the decoder can take from the encoder. The hidden state of the decoder can be
    intialized with the output from the final time-step of the encoder, and the decoder may also choose to use
    attention, in which case a weighted average of the outputs from the encoder are concatenated to the inputs
    of the decoder at every timestep.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing ``tokens`` for input tokens and ``labels`` for output tokens.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the ``EncoderDecoder`` model
    max_decoding_steps : int, required
        Length of decoded sequences
    use_attention: bool
        Should the decoder use attention to get a dynamic summary of the encoder outputs at each step of decoding?
    """
    def __init__(self, vocab: Vocabulary, source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder, max_decoding_steps: int, use_attention: bool = False) -> None:
        super(EncoderDecoder, self).__init__(vocab)
        self.source_embedder = source_embedder
        self.encoder = encoder
        self.max_decoding_steps = max_decoding_steps
        self.use_attention = use_attention
        self.start_index = self.vocab.get_token_index(START_SYMBOL, "output_tokens")
        self.end_index = self.vocab.get_token_index(END_SYMBOL, "output_tokens")
        self.num_classes = self.vocab.get_vocab_size("output_tokens")
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the hidden
        # state of the decoder with that of the final hidden states of the encoder. Also, if we're using
        # attention with ``DotProductSimilarity``, this is needed.
        self.decoder_output_dim = self.encoder.get_output_dim()
        # TODO (pradeep): target_embedding_dim need not be the same as the source embedding dim.
        target_embedding_dim = self.source_embedder.get_output_dim()
        self.target_embedder = Embedding(self.num_classes, target_embedding_dim)
        if self.use_attention:
            self.decoder_attention = Attention()
            # The output of attention will be concatenated to the input vector of the decoder at
            # each time step.
            self.decoder_input_dim = self.encoder.get_output_dim() + target_embedding_dim
        else:
            self.decoder_input_dim = target_embedding_dim
        # TODO (pradeep): Do not hardcode decoder cell type.
        self.decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)
        self.output_projection_layer = Linear(self.decoder_output_dim, self.num_classes)
        # We need the start symbol to provide as the input at the first timestep of decoding, and end symbol
        # as a way of letting decoder choose to stop decoding.

    @overrides
    def forward(self, input_tokens: Dict[str, torch.LongTensor],
                output_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        input_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the input ``TextField``. This will be passed
           through a ``TextFieldEmbedder`` and then through an encoder.
        output_tokens : Dict[str, torch.LongTensor], optional (default = None)
            Dict with indices of target side tokens, of shape ``(batch_size, target_sequence_length)``
            in ``tokens`` field.
        """
        # (batch_size, input_seq_length, enc_output_dim)
        encoder_outputs = self._get_encoder_outputs(input_tokens)
        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, enc_output_dim)
        if output_tokens:
            targets = output_tokens["tokens"]
            target_sequence_length = targets.size()[1]
            num_decoding_steps = target_sequence_length
        else:
            num_decoding_steps = self.max_decoding_steps
        # TODO (pradeep): Define a DecoderState class?
        decoder_hidden = final_encoder_output
        decoder_context = Variable(torch.zeros(1, self.decoder_output_dim))
        last_predictions = None
        step_logits = []
        step_probabilities = []
        step_predictions = []
        for timestep in range(num_decoding_steps):
            if self.training:
                input_choices = targets[:, timestep]
            else:
                if timestep == 0:
                    # For the first timestep, when we do not have targets, we input start symbols.
                    # (batch_size,)
                    input_choices = Variable(torch.LongTensor([self.start_index]).expand_as(
                            final_encoder_output[:, 0]))
                else:
                    input_choices = last_predictions
            decoder_input = self._prepare_decode_step_input(input_choices, decoder_hidden, encoder_outputs)
            decoder_hidden, decoder_context = self.decoder_cell(decoder_input,
                                                                (decoder_hidden, decoder_context))
            # (batch_size, num_classes)
            output_projections = self.output_projection_layer(decoder_hidden)
            # (batch_size, 1, num_classes)
            step_logits.append(output_projections.view(-1, 1, self.num_classes))
            decoder_predictions = self.predict_step(output_projections)
            step_probabilities.append(decoder_predictions["class_probabilities"].view(-1, 1, self.num_classes))
            last_predictions = decoder_predictions["predicted_classes"]
            # (batch_size, 1)
            step_predictions.append(last_predictions.view(-1, 1))
        # step_logits is a list containing tensors of shape (batch_size, 1, num_classes)
        # This is (batch_size, max_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)
        class_probabilities = torch.cat(step_probabilities, 1)
        all_predictions = torch.cat(step_predictions, 1)
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "predictions": all_predictions}
        if output_tokens:
            output_mask = get_text_field_mask(output_tokens)
            print(logits.size(), targets.size(), output_mask.size())
            loss = sequence_cross_entropy_with_logits(logits, targets, output_mask)
            output_dict["loss"] = loss
            # TODO: Define metrics
        return output_dict

    def _get_encoder_outputs(self, input_tokens: Dict[str, torch.LongTensor]) -> torch.LongTensor:
        embedded_input = self.source_embedder(input_tokens)
        input_mask = get_text_field_mask(input_tokens)
        encoded_text = self.encoder(embedded_input, input_mask)
        return encoded_text

    def _prepare_decode_step_input(self, input_indices: torch.LongTensor,
                                   last_decoder_output: torch.LongTensor = None,
                                   encoder_outputs: torch.LongTensor = None) -> torch.LongTensor:
        """
        Given the input indices for the current timestep of the decoder, and all the encoder outputs,
        compute the input at the current timestep.
        Note: This method is agnostic to whether the indices are gold indices or the predictions made by the
        decoder at the last timestep. So, this can be used even if we're doing some kind of scheduled sampling.

        If we're not using attention, the output of this method is just an embedding of the input indices.
        If we are, the output will be a concatentation of the embedding and an attended average of the encoder
        inputs.

        Parameters
        ----------
        input_indices : torch.LongTensor
            Indices of either the gold inputs to the decoder or the predicted labels from the previous timestep.
        last_decoder_output : torch.LongTensor, optional
            Output of from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor
            Encoder outputs from all time steps. Needed only if using attention.
        """
        # input_indices : (batch_size,)  since we are processing these one timestep at a time.
        # (batch_size, enc_embedding_dim)
        embedded_input = self.target_embedder(input_indices)
        if self.use_attention:
            if last_decoder_output is None or encoder_outputs is None:
                raise ConfigurationError("Last decoder output and all encoder outputs are needed to compute "
                                         "attention weights.")
            # encoder_outputs : (batch_size, input_sequence_length, enc_output_dim)
            # (batch_size, input_sequence_length)
            input_weights = self.decoder_attention(last_decoder_output, encoder_outputs)
            # (batch_size, enc_output_dim)
            attended_input = weighted_sum(encoder_outputs, input_weights)
            # (batch_size, enc_output_dim + enc_embedding_dim)
            return torch.cat((attended_input, embedded_input), -1)
        else:
            return embedded_input

    def predict_step(self, # pylint: disable=no-self-use
                     target_scores: torch.LongTensor) -> Dict[str, torch.LongTensor]:
        """
        Take the projected scores for all output classes and predict the output for a single timestep.
        Here we simply normalize the scores and output the class with the highest probability as the
        prediction. This method can be overridden to do a constrained prediction and/or avoid local
        normalization.

        Parameters
        ----------
        target_scores : torch.LongTensor
            Unnormalized class scores, the output from a projection layer. (batch_size, num_classes)

        Returns
        -------
        prediction_dict : Dict[str, torch.LongTensor] with fields
            class_probabilities
            predicted_classes
        """
        # (batch_size, num_classes)
        class_probabilities = F.softmax(target_scores)
        # (batch_size,)
        _, predicted_classes = torch.max(class_probabilities, 1)
        prediction_dict = {"class_probabilities": class_probabilities, "predicted_classes": predicted_classes}
        return prediction_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Trims the output predictions to the first end symbol, replaces indics with corresponding tokens,
        and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            indices = indices[:indices.index(self.end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace="output_tokens")
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        if len(all_predicted_tokens) == 1:
            all_predicted_tokens = all_predicted_tokens[0]
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'EncoderDecoder':
        source_embedder_params = params.pop("source_embedder")
        source_embedder = TextFieldEmbedder.from_params(vocab, source_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        use_attention = params.pop("use_attention")
        return cls(vocab, source_embedder, encoder, max_decoding_steps, use_attention)
