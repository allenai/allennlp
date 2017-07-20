import torch

from allennlp.experiments import Registry
from allennlp.modules.seq2seq_encoders.wrapped_pytorch_rnn import WrappedPytorchRnn

# pylint: disable=protected-access
Registry.register_seq2seq_encoder("gru")(WrappedPytorchRnn._Wrapper(torch.nn.GRU))
Registry.register_seq2seq_encoder("lstm")(WrappedPytorchRnn._Wrapper(torch.nn.LSTM))
Registry.register_seq2seq_encoder("rnn")(WrappedPytorchRnn._Wrapper(torch.nn.RNN))
