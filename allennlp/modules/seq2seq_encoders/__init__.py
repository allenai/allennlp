"""
Modules that transform a sequence of input vectors
into a sequence of output vectors.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.

The available Seq2Seq encoders are

* `"gru" <https://pytorch.org/docs/master/nn.html#torch.nn.GRU>`_
* `"lstm" <https://pytorch.org/docs/master/nn.html#torch.nn.LSTM>`_
* `"rnn" <https://pytorch.org/docs/master/nn.html#torch.nn.RNN>`_
* :class:`"augmented_lstm" <allennlp.modules.augmented_lstm.AugmentedLstm>`
* :class:`"alternating_lstm" <allennlp.modules.stacked_alternating_lstm.StackedAlternatingLstm>`
* :class:`"alternating_highway_lstm" <allennlp.modules.stacked_alternating_lstm.StackedAlternatingLstm> (GPU only)`
* :class:`"stacked_self_attention" <allennlp.modules.stacked_self_attention.StackedSelfAttentionEncoder>`
* :class:`"multi_head_self_attention" <allennlp.modules.multi_head_self_attention.MultiHeadSelfAttention>`
* :class:`"pass_through" <allennlp.modules.pass_through_encoder.PassThroughEncoder>`
* :class:`"feedforward" <allennlp.modules.feedforward_encoder.FeedforwardEncoder>`
"""

from typing import Type
import logging

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import \
        BidirectionalLanguageModelTransformer
from allennlp.modules.seq2seq_encoders.gated_cnn_encoder import GatedCnnEncoder
from allennlp.modules.seq2seq_encoders.intra_sentence_attention import IntraSentenceAttentionEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.modules.seq2seq_encoders.pass_through_encoder import PassThroughEncoder
from allennlp.modules.seq2seq_encoders.feedforward_encoder import FeedForwardEncoder
from allennlp.modules.seq2seq_encoders.qanet_encoder import QaNetEncoder


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class _Seq2SeqWrapper:
    """
    For :class:`Registrable` we need to have a ``Type[Seq2SeqEncoder]`` as the value registered for each
    key.  What that means is that we need to be able to ``__call__`` these values (as is done with
    ``__init__`` on the class), and be able to call ``from_params()`` on the value.

    In order to accomplish this, we have two options: (1) we create a ``Seq2SeqEncoder`` class for
    all of pytorch's RNN modules individually, with our own parallel classes that we register in
    the registry; or (2) we wrap pytorch's RNNs with something that `mimics` the required
    API.  We've gone with the second option here.

    This is a two-step approach: first, we have the :class:`PytorchSeq2SeqWrapper` class that handles
    the interface between a pytorch RNN and our ``Seq2SeqEncoder`` API.  Our ``PytorchSeq2SeqWrapper``
    takes an instantiated pytorch RNN and just does some interface changes.  Second, we need a way
    to create one of these ``PytorchSeq2SeqWrappers``, with an instantiated pytorch RNN, from the
    registry.  That's what this ``_Wrapper`` does.  The only thing this class does is instantiate
    the pytorch RNN in a way that's compatible with ``Registrable``, then pass it off to the
    ``PytorchSeq2SeqWrapper`` class.

    When you instantiate a ``_Wrapper`` object, you give it an ``RNNBase`` subclass, which we save
    to ``self``.  Then when called (as if we were instantiating an actual encoder with
    ``Encoder(**params)``, or with ``Encoder.from_params(params)``), we pass those parameters
    through to the ``RNNBase`` constructor, then pass the instantiated pytorch RNN to the
    ``PytorchSeq2SeqWrapper``.  This lets us use this class in the registry and have everything just
    work.
    """
    PYTORCH_MODELS = [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]

    def __init__(self, module_class: Type[torch.nn.modules.RNNBase]) -> None:
        self._module_class = module_class

    def __call__(self, **kwargs) -> PytorchSeq2SeqWrapper:
        return self.from_params(Params(kwargs))

    # Logic requires custom from_params
    def from_params(self, params: Params) -> PytorchSeq2SeqWrapper:
        if not params.pop_bool('batch_first', True):
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        if self._module_class in self.PYTORCH_MODELS:
            params['batch_first'] = True
        stateful = params.pop_bool('stateful', False)
        module = self._module_class(**params.as_dict(infer_type_and_cast=True))
        return PytorchSeq2SeqWrapper(module, stateful=stateful)

# pylint: disable=protected-access
Seq2SeqEncoder.register("gru")(_Seq2SeqWrapper(torch.nn.GRU))
Seq2SeqEncoder.register("lstm")(_Seq2SeqWrapper(torch.nn.LSTM))
Seq2SeqEncoder.register("rnn")(_Seq2SeqWrapper(torch.nn.RNN))
Seq2SeqEncoder.register("augmented_lstm")(_Seq2SeqWrapper(AugmentedLstm))
Seq2SeqEncoder.register("alternating_lstm")(_Seq2SeqWrapper(StackedAlternatingLstm))
Seq2SeqEncoder.register("stacked_bidirectional_lstm")(_Seq2SeqWrapper(StackedBidirectionalLstm))
