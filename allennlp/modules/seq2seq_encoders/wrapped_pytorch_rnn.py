import torch

from allennlp.modules import Seq2SeqEncoder


class WrappedPytorchRnn(Seq2SeqEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the first one as a single output.  This
    wrapper pulls out that output, and adds a :func:`get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from ``get_output_dim``.
    """
    def __init__(self, module: torch.nn.modules.RNNBase) -> None:
        super(WrappedPytorchRnn, self).__init__()
        self._module = module

    def get_output_dim(self) -> int:
        return self._module.hidden_size * (2 if self._module.bidirectional else 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        return self._module(inputs)[0]
