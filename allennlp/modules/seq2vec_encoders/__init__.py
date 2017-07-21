from typing import Type

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.experiments import Registry
from allennlp.modules.seq2vec_encoders.wrapped_pytorch_rnn import WrappedPytorchRnn

class _Wrapper:
    """
    The :class:`Registry` needs to have a ``Type[Seq2VecEncoder]`` as the value registered for each
    key.  What that means is that we need to be able to ``__call__`` these values (as is done with
    ``__init__`` on the class), and be able to call ``from_params()`` on the value.

    In order to accomplish this, we have two options: (1) we create a ``Seq2VecEncoder`` class for
    all of pytorch's RNN modules individually, with our own parallel classes that we register in
    the registry; or (2) we wrap pytorch's RNNs with something that `mimics` the required
    ``Registry`` API.  We've gone with the second option here.

    This is a two-step approach: first, we have the :class:`WrappedPytorchRnn` class that handles
    the interface between a pytorch RNN and our ``Seq2VecEncoder`` API.  Our ``WrappedPytorchRnn``
    takes an instantiated pytorch RNN and just does some interface changes.  Second, we need a way
    to create one of these ``WrappedPytorchRnns``, with an instantiated pytorch RNN, from the
    registry.  That's what this ``_Wrapper`` does.  The only thing this class does is instantiate
    the pytorch RNN in a way that's compatible with the ``Registry``, then pass it off to the
    ``WrappedPytorchRnn`` class.

    When you instantiate a ``_Wrapper`` object, you give it an ``RNNBase`` subclass, which we save
    to ``self``.  Then when called (as if we were instantiating an actual encoder with
    ``Encoder(**params)``, or with ``Encoder.from_params(params)``), we pass those parameters
    through to the ``RNNBase`` constructor, then pass the instantiated pytorch RNN to the
    ``WrappedPytorchRnn``.  This lets us use this class in the registry and have everything just
    work.
    """
    def __init__(self, module_class: Type[torch.nn.modules.RNNBase]) -> None:
        self._module_class = module_class

    def __call__(self, **kwargs) -> WrappedPytorchRnn:
        if not kwargs.pop('batch_first', True):
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        kwargs['batch_first'] = True
        module = self._module_class(**kwargs)
        return WrappedPytorchRnn(module)

    def from_params(self, params: Params) -> WrappedPytorchRnn:
        return self(**params.as_dict())

# pylint: disable=protected-access
Registry.register_seq2vec_encoder("gru")(WrappedPytorchRnn._Wrapper(torch.nn.GRU))
Registry.register_seq2vec_encoder("lstm")(WrappedPytorchRnn._Wrapper(torch.nn.LSTM))
Registry.register_seq2vec_encoder("rnn")(WrappedPytorchRnn._Wrapper(torch.nn.RNN))
