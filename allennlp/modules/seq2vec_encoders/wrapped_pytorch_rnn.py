import torch

from allennlp.modules import Seq2VecEncoder


class WrappedPytorchRnn(Seq2VecEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the second one as a single output.  This
    wrapper pulls out that output, and adds a :func:`get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from ``get_output_dim``.

    Also, there are lots of ways you could imagine going from an RNN hidden state at every
    timestep to a single vector - you could take the last vector at all layers in the stack, do
    some kind of pooling, take the last vector of the top layer in a stack, or many other  options.
    We just take the final hidden state vector.  TODO(mattg): allow for other ways of wrapping
    RNNs.
    """
    def __init__(self, module: torch.nn.modules.RNNBase) -> None:
        super(WrappedPytorchRnn, self).__init__()
        self._module = module

    def get_output_dim(self) -> int:
        return self._module.hidden_size * self._module.num_directions

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        return self._module(inputs)[0][-1]
