from typing import Type

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.experiments import Registry
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper


class _Seq2VecWrapper:
    """
    The :class:`Registry` needs to have a ``Type[Seq2VecEncoder]`` as the value registered for each
    key.  What that means is that we need to be able to ``__call__`` these values (as is done with
    ``__init__`` on the class), and be able to call ``from_params()`` on the value.

    In order to accomplish this, we have two options: (1) we create a ``Seq2VecEncoder`` class for
    all of pytorch's RNN modules individually, with our own parallel classes that we register in
    the registry; or (2) we wrap pytorch's RNNs with something that `mimics` the required
    ``Registry`` API.  We've gone with the second option here.

    This is a two-step approach: first, we have the :class:`PytorchSeq2VecWrapper` class that handles
    the interface between a pytorch RNN and our ``Seq2VecEncoder`` API.  Our ``PytorchSeq2VecWrapper``
    takes an instantiated pytorch RNN and just does some interface changes.  Second, we need a way
    to create one of these ``PytorchSeq2VecWrappers``, with an instantiated pytorch RNN, from the
    registry.  That's what this ``_Wrapper`` does.  The only thing this class does is instantiate
    the pytorch RNN in a way that's compatible with the ``Registry``, then pass it off to the
    ``PytorchSeq2VecWrapper`` class.

    When you instantiate a ``_Wrapper`` object, you give it an ``RNNBase`` subclass, which we save
    to ``self``.  Then when called (as if we were instantiating an actual encoder with
    ``Encoder(**params)``, or with ``Encoder.from_params(params)``), we pass those parameters
    through to the ``RNNBase`` constructor, then pass the instantiated pytorch RNN to the
    ``PytorchSeq2VecWrapper``.  This lets us use this class in the registry and have everything just
    work.
    """
    def __init__(self, module_class: Type[torch.nn.modules.RNNBase]) -> None:
        self._module_class = module_class

    def __call__(self, **kwargs) -> PytorchSeq2VecWrapper:
        return self.from_params(Params(kwargs))

    def from_params(self, params: Params) -> PytorchSeq2VecWrapper:
        if not params.pop('batch_first', True):
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        params['batch_first'] = True
        module = self._module_class(**params.as_dict())
        return PytorchSeq2VecWrapper(module)

# pylint: disable=protected-access
Registry.register_seq2vec_encoder("gru")(_Seq2VecWrapper(torch.nn.GRU))
Registry.register_seq2vec_encoder("lstm")(_Seq2VecWrapper(torch.nn.LSTM))
Registry.register_seq2vec_encoder("rnn")(_Seq2VecWrapper(torch.nn.RNN))
