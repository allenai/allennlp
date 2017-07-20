from typing import Type

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules import Seq2VecEncoder


class WrappedPytorchRnn(Seq2VecEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the second one as a single output.  This
    wrapper pulls out that output, and allows us to instantiate the pytorch RNN from parameters.

    This is actually slightly more complicated than that, because the LSTM's second output is a
    tuple of (hidden state, cell state), and we just return the hidden state.

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

    class _Wrapper:
        """
        This wrapper gives us something with a ``__call__`` method and a ``from_params`` method
        that we can use to instantiate a ``WrappedPytorchRnn``.  We use this when registering
        built-in pytorch RNNs in the registry.
        """
        def __init__(self, module_class: Type[torch.nn.Module]) -> None:
            self._module_class = module_class

        def __call__(self, **kwargs) -> 'WrappedPytorchRnn':
            if not kwargs.pop('batch_first', True):
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
            kwargs['batch_first'] = True
            module = self._module_class(**kwargs)
            return WrappedPytorchRnn(module)

        def from_params(self, params: Params) -> 'WrappedPytorchRnn':
            return self(**params.as_dict())
