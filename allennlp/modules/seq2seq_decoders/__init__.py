"""
Modules that transform a sequence of encoded vectors
into a sequence of output vectors.

The available Seq2Seq decoders are

* :class:`"simple_decoder" <allennlp.modules.seq2seq_decoders.lstm_decoder_cell.LstmDecoderCell>`

"""
from allennlp.modules.seq2seq_decoders.lstm_decoder_cell import LstmDecoderCell
from allennlp.modules.seq2seq_decoders.rnn_seq_decoder import RnnSeqDecoder
