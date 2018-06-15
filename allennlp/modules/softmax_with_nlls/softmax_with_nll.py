from overrides import overrides

import torch
import torch.nn as nn
from torch import Tensor as Tensor
import torch.nn.functional as F
from torch.autograd import Variable

from allennlp.common.checks import check_dimensions_match
from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
from allennlp.data import Vocabulary

class SoftmaxWithNLL(torch.nn.Module, Registrable):
    """
    Wrapper of RNN used in LMs
    """
    @overrides
    def forward(self, x, target) -> Tensor:
        raise NotImplementedError

    def log_prob(self) -> Tensor:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SoftmaxWithNLL':
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(vocab, params)

