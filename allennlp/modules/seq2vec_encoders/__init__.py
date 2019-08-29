"""
Modules that transform a sequence of input vectors
into a single output vector.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.

The available Seq2Vec encoders are

* `"gru" <https://pytorch.org/docs/master/nn.html#torch.nn.GRU>`_
* `"lstm" <https://pytorch.org/docs/master/nn.html#torch.nn.LSTM>`_
* `"rnn" <https://pytorch.org/docs/master/nn.html#torch.nn.RNN>`_
* :class:`"cnn" <allennlp.modules.seq2vec_encoders.cnn_encoder.CnnEncoder>`
* :class:`"augmented_lstm" <allennlp.modules.augmented_lstm.AugmentedLstm>`
* :class:`"alternating_lstm" <allennlp.modules.stacked_alternating_lstm.StackedAlternatingLstm>`
* :class:`"stacked_bidirectional_lstm" <allennlp.modules.stacked_bidirectional_lstm.StackedBidirectionalLstm>`
"""

from typing import Type

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2vec_encoders.cnn_highway_encoder import CnnHighwayEncoder
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm


class _Seq2VecWrapper:
    """
    For :class:`Registrable` we need to have a ``Type[Seq2VecEncoder]`` as the value registered for each
    key.  What that means is that we need to be able to ``__call__`` these values (as is done with
    ``__init__`` on the class), and be able to call ``from_params()`` on the value.

    In order to accomplish this, we have two options: (1) we create a ``Seq2VecEncoder`` class for
    all of pytorch's RNN modules individually, with our own parallel classes that we register in
    the registry; or (2) we wrap pytorch's RNNs with something that `mimics` the required
    API.  We've gone with the second option here.

    This is a two-step approach: first, we have the :class:`PytorchSeq2VecWrapper` class that handles
    the interface between a pytorch RNN and our ``Seq2VecEncoder`` API.  Our ``PytorchSeq2VecWrapper``
    takes an instantiated pytorch RNN and just does some interface changes.  Second, we need a way
    to create one of these ``PytorchSeq2VecWrappers``, with an instantiated pytorch RNN, from the
    registry.  That's what this ``_Wrapper`` does.  The only thing this class does is instantiate
    the pytorch RNN in a way that's compatible with ``Registrable``, then pass it off to the
    ``PytorchSeq2VecWrapper`` class.

    When you instantiate a ``_Wrapper`` object, you give it an ``RNNBase`` subclass, which we save
    to ``self``.  Then when called (as if we were instantiating an actual encoder with
    ``Encoder(**params)``, or with ``Encoder.from_params(params)``), we pass those parameters
    through to the ``RNNBase`` constructor, then pass the instantiated pytorch RNN to the
    ``PytorchSeq2VecWrapper``.  This lets us use this class in the registry and have everything just
    work.
    """
    PYTORCH_MODELS = [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]
    def __init__(self, module_class: Type[torch.nn.modules.RNNBase]) -> None:
        self._module_class = module_class

    def __call__(self, **kwargs) -> PytorchSeq2VecWrapper:
        return self.from_params(Params(kwargs))

    # Logic requires custom from_params
    def from_params(self, params: Params) -> PytorchSeq2VecWrapper:
        if not params.pop('batch_first', True):
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        if self._module_class in self.PYTORCH_MODELS:
            params['batch_first'] = True
        module = self._module_class(**params.as_dict(infer_type_and_cast=True))
        return PytorchSeq2VecWrapper(module)

# pylint: disable=protected-access
Seq2VecEncoder.register("gru")(_Seq2VecWrapper(torch.nn.GRU))
Seq2VecEncoder.register("lstm")(_Seq2VecWrapper(torch.nn.LSTM))
Seq2VecEncoder.register("rnn")(_Seq2VecWrapper(torch.nn.RNN))
Seq2VecEncoder.register("augmented_lstm")(_Seq2VecWrapper(AugmentedLstm))
Seq2VecEncoder.register("alternating_lstm")(_Seq2VecWrapper(StackedAlternatingLstm))
Seq2VecEncoder.register("stacked_bidirectional_lstm")(_Seq2VecWrapper(StackedBidirectionalLstm))
