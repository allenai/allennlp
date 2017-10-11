from typing import Dict, List, Tuple, Type

import numpy
from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.nn.decoding import DecodingAlgorithm, DecoderState


@Model.register("wikitables_parser")
class WikiTablesSemanticParser(Model):
    """
    A ``WikiTablesSemanticParser`` is a :class:`Model` which takes as input a table and a question,
    and produces a logical form that answers the question when executed over the table.  The
    logical form is generated by a `type-constrained`, `transition-based` parser.  This is a
    re-implementation of the model used for the paper `Neural Semantic Parsing with Type
    Constraints for Semi-Structured Tables
    <https://www.semanticscholar.org/paper/Neural-Semantic-Parsing-with-Type-Constraints-for-Krishnamurthy-Dasigi/8c6f58ed0ebf379858c0bbe02c53ee51b3eb398a>`_,
    by Jayant Krishnamurthy, Pradeep Dasigi, and Matt Gardner (EMNLP 2017).

    WORK STILL IN PROGRESS.  This is just copying the SimpleSeq2Seq model for now, and we'll
    iteratively improve it until we've reproduced the performance of the original parser.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder to use for the input question
    max_decoding_steps : int, required
        Length of decoded sequences
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention_function: ``SimilarityFunction``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: DecodingAlgorithm,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 attention_function: SimilarityFunction = None) -> None:
        super(WikiTablesSemanticParser, self).__init__(vocab)
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._decoder = decoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()
        self._attention_function = attention_function

        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder. Also, if
        # we're using attention with ``DotProductSimilarity``, this is needed.
        self._decoder_output_dim = self._encoder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)
        if self._attention_function:
            self._decoder_attention = Attention(self._attention_function)
            # The output of attention, a weighted average over encoder outputs, will be
            # concatenated to the input vector of the decoder at each time step.
            self._decoder_input_dim = self._encoder.get_output_dim() + target_embedding_dim
        else:
            self._decoder_input_dim = target_embedding_dim
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        embedded_input = self._source_embedder(source_tokens)
        source_mask = util.get_text_field_mask(source_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)

        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        with torch.cuda.device_of(final_encoder_output):
            decoder_context = Variable(torch.zeros(1, self._decoder_output_dim))

        if target_tokens:
            targets = target_tokens["tokens"]
            target_mask = util.get_text_field_mask(target_tokens)
            target_sequence_length = targets.size()[1]
            # The last input from the target is either padding or the end symbol. Either way, we
            # don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            targets = None
            num_decoding_steps = self._max_decoding_steps

        initial_state = DecoderState(encoder_outputs=encoder_outputs,
                                     encoder_output_mask=source_mask.float(),
                                     hidden_state=(final_encoder_output, decoder_context))
        return self._decoder.decode(num_decoding_steps,
                                    initial_state,
                                    self._decode_step,
                                    self.training,
                                    targets,
                                    target_mask)

    def _decode_step(self,
                     decoder_state: DecoderState,
                     step_input: torch.Tensor) -> Tuple[torch.Tensor, List[int], Tuple[torch.Tensor, torch.Tensor]]:
        embedded_input = self._target_embedder(step_input)
        decoder_hidden, decoder_context = decoder_state.hidden_state
        if self._attention_function:
            # (batch_size, input_sequence_length)
            input_weights = self._decoder_attention(decoder_hidden,
                                                    decoder_state.encoder_outputs,
                                                    decoder_state.encoder_output_mask)
            # (batch_size, encoder_output_dim)
            attended_input = util.weighted_sum(decoder_state.encoder_outputs, input_weights)
            # (batch_size, encoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            decoder_input = embedded_input
        decoder_hidden, decoder_context = self._decoder_cell(decoder_input,
                                                             (decoder_hidden, decoder_context))
        # (batch_size, num_classes)
        logits = self._output_projection_layer(decoder_hidden)
        with torch.cuda.device_of(logits):
            output_mask = decoder_state.get_output_mask()
        normalized_logits = util.masked_log_softmax(logits, output_mask)
        return normalized_logits, decoder_state.get_valid_actions(), (decoder_hidden, decoder_context)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.data.cpu().numpy()
        all_predicted_tokens = []
        end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            if end_index in indices:
                indices = indices[:indices.index(end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace="target_tokens")
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        if len(all_predicted_tokens) == 1:
            all_predicted_tokens = all_predicted_tokens[0]  # type: ignore
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'WikiTablesSemanticParser':
        source_embedder_params = params.pop("source_embedder")
        source_embedder = TextFieldEmbedder.from_params(vocab, source_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        target_namespace = params.pop("target_namespace", "tokens")
        target_embedding_dim = params.pop("target_embedding_dim", None)
        decoder = DecodingAlgorithm.from_params(vocab, target_namespace, params.pop("decoder"))
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        scheduled_sampling_ratio = params.pop("scheduled_sampling_ratio", 0.0)
        return cls(vocab,
                   source_embedder=source_embedder,
                   encoder=encoder,
                   decoder=decoder,
                   max_decoding_steps=max_decoding_steps,
                   target_namespace=target_namespace,
                   target_embedding_dim=target_embedding_dim,
                   attention_function=attention_function)
