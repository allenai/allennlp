import torch

from allennlp.experiments import Registry
from allennlp.modules.seq2vec_encoders.wrapped_pytorch_rnn import WrappedPytorchRnn

# pylint: disable=protected-access
Registry.register_seq2vec_encoder("gru")(WrappedPytorchRnn._Wrapper(torch.nn.GRU))
Registry.register_seq2vec_encoder("lstm")(WrappedPytorchRnn._Wrapper(torch.nn.LSTM))
Registry.register_seq2vec_encoder("rnn")(WrappedPytorchRnn._Wrapper(torch.nn.RNN))
