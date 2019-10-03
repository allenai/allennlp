"""
Modules that transform a sequence of encoded vectors
into a sequence of output vectors.

The available Seq2Seq decoders are

* :class:`"auto_regressive_seq_decoder"
  <allennlp.modules.seq2seq_decoders.auto_regressive_seq_decoder.AutoRegressiveSeqDecoder>`

"""
from allennlp.modules.seq2seq_decoders.decoder_net import DecoderNet
from allennlp.modules.seq2seq_decoders.lstm_cell_decoder_net import LstmCellDecoderNet
from allennlp.modules.seq2seq_decoders.stacked_self_attention_decoder_net import (
    StackedSelfAttentionDecoderNet,
)
from allennlp.modules.seq2seq_decoders.seq_decoder import SeqDecoder
from allennlp.modules.seq2seq_decoders.auto_regressive_seq_decoder import AutoRegressiveSeqDecoder
