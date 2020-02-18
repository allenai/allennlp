"""
Modules that transform a sequence of input vectors
into a single output vector.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.

The available Seq2Vec encoders are

* `"gru"` https://pytorch.org/docs/master/nn.html#torch.nn.GRU
* `"lstm"` https://pytorch.org/docs/master/nn.html#torch.nn.LSTM
* `"rnn"` https://pytorch.org/docs/master/nn.html#torch.nn.RNN
* `"cnn"` allennlp.modules.seq2vec_encoders.cnn_encoder.CnnEncoder
* `"augmented_lstm"` allennlp.modules.augmented_lstm.AugmentedLstm
* `"alternating_lstm"` allennlp.modules.stacked_alternating_lstm.StackedAlternatingLstm
* `"stacked_bidirectional_lstm"` allennlp.modules.stacked_bidirectional_lstm.StackedBidirectionalLstm
"""

from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders.cls_pooler import ClsPooler
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2vec_encoders.cnn_highway_encoder import CnnHighwayEncoder
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import (
    AugmentedLstmSeq2VecEncoder,
    GruSeq2VecEncoder,
    LstmSeq2VecEncoder,
    PytorchSeq2VecWrapper,
    RnnSeq2VecEncoder,
    StackedAlternatingLstmSeq2VecEncoder,
    StackedBidirectionalLstmSeq2VecEncoder,
)
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
