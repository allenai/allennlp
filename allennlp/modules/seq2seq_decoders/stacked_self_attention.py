from typing import Dict, List, Tuple

import torch
import copy
import math
import torch.nn.functional as F
from torch.nn import Linear
from torch.autograd import Variable

from allennlp.common import Registrable
from allennlp.modules.seq2seq_decoders.seq_decoder import SeqDecoder
from allennlp.data import Vocabulary
from allennlp.modules import TokenEmbedder
from allennlp.modules.seq2seq_decoders.decoder_cell import DecoderCell
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric
import allennlp.modules.token_embedders

@SeqDecoder.register("stacked_self_attention_seq_decoder")
class StackedSelfAttentionSeqDecoder(SeqDecoder):
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
    """

    def __init__(
            self,
            vocab: Vocabulary,
            hidden_dim: int,
            feedforward_hidden_dim: int,
            num_layers: int,
            num_attention_heads: int,
            dropout_prob: float = 0.1,
            residual_dropout_prob: float = 0.2,
            attention_dropout_prob: float = 0.1,
            max_decoding_steps: int,
            target_embedder: TokenEmbedder,
            target_namespace: str = "tokens",
            bidirectional_input: bool = False,
            beam_size: int = None,
            tensor_based_metric: Metric = None,
            token_based_metric: Metric = None,
    ):

        super(StackedSelfAttentionSeqDecoder, self).__init__(
            vocab=vocab,
            target_namespace=target_namespace,
            tensor_based_metric=tensor_based_metric,
            token_based_metric=token_based_metric
        )

        self.bidirectional_input = bidirectional_input


        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)


        c = copy.deepcopy
        attn = MultiHeadedAttention(num_attention_heads, hidden_dim, attention_dropout_prob)
        ff = PositionwiseFeedForward(hidden_dim, feedforward_hidden_dim, dropout_prob)
        self._embed_scale = math.sqrt(hidden_dim)
        self._positional_embedder = PositionalEncoding(hidden_dim, dropout_prob)
        self._decoder = Decoder(DecoderLayer(hidden_dim, c(attn), c(attn),
                                             ff, residual_dropout_prob), num_layers)

        target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)
        # TODO Add option for tying embeddings
        self._generator = Generator(hidden_dim, self._target_vocab_size)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.decoder_output_dim = hidden_dim
        self.decoder_input_dim = self.decoder_output_dim
        self.encoder_output_dim = self.decoder_input_dim

        self._target_embedder = target_embedder
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

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

        # shape: (group_size, 1, max_input_sequence_length)
        source_mask = state["source_mask"].unsqueeze(-2)

        # shape: (group_size, steps_count, decoder_output_dim)
        previous_steps_predictions = state.get("previous_steps_predictions")

        # shape: (batch_size, 1, target_embedding_dim)
        last_predictions_embeddings = self._target_embedder(last_predictions) * self._embed_scale
        last_predictions_embeddings = last_predictions_embeddings.unsqueeze(1)

        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in ``last_predictions``
            # shape: (group_size, 1, target_embedding_dim)
            previous_steps_predictions = last_predictions_embeddings
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat([previous_steps_predictions, last_predictions_embeddings], 1)

        state["previous_steps_predictions"] = previous_steps_predictions


        last_predictions_embeddings_with_pos = self._positional_embedder(last_predictions_embeddings)

        # shape: (group_size, last_predictions_time_steps + 1, hidden_dim)
        target_encoded = self._decoder(last_predictions_embeddings_with_pos, encoder_outputs, source_mask, None)

        # shape: (group_size, hidden_dim)
        last_target_encoded = target_encoded[:,-1,:]


        # shape: (group_size, num_classes)
        output_projections = self._generator(last_target_encoded)

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

    def get_output_dim(self):
        return self.decoder_output_dim

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
        output_dict = {}
        if target_tokens:
            state = encoder_out
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            source_mask = state["source_mask"].unsqueeze(-2)
            encoder_outputs = state["encoder_outputs"]

            # shape: (batch_size, max_target_batch_sequence_length)
            original_targets = target_tokens["tokens"]
            # shape: (batch_size, max_target_batch_sequence_length)
            original_target_mask = util.get_text_field_mask(target_tokens)

            # Leaving out last EOS (but guess this is imperfect)
            targets = original_targets[:,:-1]
            target_mask = original_target_mask[:,:-1]

            target_mask = target_mask.unsqueeze(-2)
            target_mask_without_future = target_mask & Variable(
                subsequent_mask(targets.size(-1)).type_as(target_mask.data))
            target_embedding = self._target_embedder(targets) * self._embed_scale
            target_embedding = self._positional_embedder(target_embedding)
            target_encoded = self._decoder(target_embedding, encoder_outputs, source_mask, target_mask_without_future)
            target_logits = self._generator(target_encoded)
            loss = self._get_loss(target_logits, original_targets, original_target_mask)
            output_dict["loss"] = loss

        if not self.training:
            state = encoder_out
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



from torch import nn
import torch
import numpy as np

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
