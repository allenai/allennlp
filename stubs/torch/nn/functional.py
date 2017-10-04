from typing import Union, Generic, TypeVar, Optional, overload

from torch.tensor import _TensorBase
from torch.autograd import Variable

T = TypeVar('T', '_TensorBase', 'Variable')

def softmax(inputs: T) -> T: ...
def log_softmax(inputs: T) -> T: ...
def sigmoid(inputs: T) -> T: ...

def nll_loss(input: T,
             target: T,
             weight: Optional['Variable'] = None,
             size_average: Optional[bool] = None,
             ignore_index: Optional[int] = None) -> T: ...

def embedding(input: T,
              embedding_matrix: T,
              max_norm: Optional[float] = None,
              norm_type: float = 2,
              scale_grad_by_freq: bool = False,
              sparse: bool = False) -> T: ...

def relu(inputs: Variable) -> Variable: ...
