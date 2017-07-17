from typing import Dict, Optional, Union

import torch
from torch.autograd import Variable
import numpy

ArrayOrDictOfArrays = Union[numpy.array, Dict[str, numpy.array]]
RecursiveDictOfArrays = Union[ArrayOrDictOfArrays, Dict[str, ArrayOrDictOfArrays]]


def data_structure_as_variables(data_structure: RecursiveDictOfArrays,
                                cuda_device: Optional[int] = -1):

    if isinstance(data_structure, dict):
        for key, value in data_structure.items():
            data_structure[key] = data_structure_as_variables(value)
        return data_structure
    else:

        torch_variable = Variable(torch.from_numpy(data_structure))
        if cuda_device == -1:
            return torch_variable
        else:
            return torch_variable.cuda(cuda_device)
