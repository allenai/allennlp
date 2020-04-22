"""
Modules that transform a sequence of input vectors
into a sequence of output vectors.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.

The available Seq2Seq encoders are

- `"gru"` : https://pytorch.org/docs/master/nn.html#torch.nn.GRU
- `"lstm"` : https://pytorch.org/docs/master/nn.html#torch.nn.LSTM
- `"rnn"` : https://pytorch.org/docs/master/nn.html#torch.nn.RNN
- `"augmented_lstm"` : allennlp.modules.augmented_lstm.AugmentedLstm
- `"alternating_lstm"` : allennlp.modules.stacked_alternating_lstm.StackedAlternatingLstm
- `"alternating_highway_lstm"` : allennlp.modules.stacked_alternating_lstm.StackedAlternatingLstm (GPU only)
- `"stacked_self_attention"` : allennlp.modules.stacked_self_attention.StackedSelfAttentionEncoder
- `"multi_head_self_attention"` : allennlp.modules.multi_head_self_attention.MultiHeadSelfAttention
- `"pass_through"` : allennlp.modules.pass_through_encoder.PassThroughEncoder
- `"feedforward"` : allennlp.modules.feedforward_encoder.FeedforwardEncoder
- `"pytorch_transformer"` : allennlp.modules.seq2seq_encoders.PytorchTransformer
"""

from allennlp.modules.seq2seq_encoders.compose_encoder import ComposeEncoder
from allennlp.modules.seq2seq_encoders.feedforward_encoder import FeedForwardEncoder
from allennlp.modules.seq2seq_encoders.gated_cnn_encoder import GatedCnnEncoder
from allennlp.modules.seq2seq_encoders.pass_through_encoder import PassThroughEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import (
    AugmentedLstmSeq2SeqEncoder,
    GruSeq2SeqEncoder,
    LstmSeq2SeqEncoder,
    PytorchSeq2SeqWrapper,
    RnnSeq2SeqEncoder,
    StackedAlternatingLstmSeq2SeqEncoder,
    StackedBidirectionalLstmSeq2SeqEncoder,
)
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders.pytorch_transformer_wrapper import PytorchTransformer
