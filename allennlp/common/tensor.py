from typing import Dict, Union

import torch
from torch.autograd import Variable
import numpy


def arrays_to_variables(data_structure: Dict[str, Union[dict, numpy.ndarray]],
                        cuda_device: int = -1):
    """
    Convert an (optionally) nested dictionary of arrays to Pytorch ``Variables``,
    suitable for use in a computation graph.
    """
    if isinstance(data_structure, dict):
        for key, value in data_structure.items():
            data_structure[key] = arrays_to_variables(value)
        return data_structure
    else:
        torch_variable = Variable(torch.from_numpy(data_structure))
        if cuda_device == -1:
            return torch_variable
        else:
            return torch_variable.cuda(cuda_device)
