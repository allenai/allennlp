"""
Modules that transform a sequence of encoded vectors
into a sequence of output vectors.

The available Seq2Seq decoders are

* :class:`"simple_decoder" <allennlp.modules.seq2seq_decoders.lstm_decoder_cell.LstmDecoderCell>`

"""
from allennlp.modules.seq2seq_decoders.decoder_module import DecoderModule
from allennlp.modules.seq2seq_decoders.lstm_cell_module import LstmCellModule
from allennlp.modules.seq2seq_decoders.stacked_self_attention_module import StackedSelfAttentionDecoderModule
from allennlp.modules.seq2seq_decoders.seq_decoder import SeqDecoder
from allennlp.modules.seq2seq_decoders.default_seq_decoder import DefaultSeqDecoder
