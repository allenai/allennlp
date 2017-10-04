import torch
from torch.autograd import Variable

from allennlp.common import Registrable

class Regularizer(Registrable):
    """
    An abstract class representing a regularizer. It must implement
    call, returning a scalar tensor.
    """
    default_implementation = 'l2'

    def __call__(self, parameter: Variable) -> Variable:
        raise NotImplementedError
