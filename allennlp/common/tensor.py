from typing import Dict, Optional, Union

import torch
from torch.autograd import Variable
import numpy

DictOfArrays = Dict[str, Union['DictOfArrays', numpy.ndarray]]  # pylint: disable=invalid-name


def data_structure_as_variables(data_structure: DictOfArrays,
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
