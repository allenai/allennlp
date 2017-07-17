from typing import Dict, Optional, Union

import torch
import numpy

ArrayOrDictOfArrays = Union[numpy.array, Dict[str, numpy.array]]
RecursiveDictOfArrays = Union[ArrayOrDictOfArrays, Dict[str, ArrayOrDictOfArrays]]


def data_structure_as_tensors(data_structure: RecursiveDictOfArrays,
                              cuda_device: Optional[int] = -1):

    if isinstance(data_structure, dict):
        for key, value in data_structure.items():
            data_structure[key] = data_structure_as_tensors(value)
        return data_structure
    else:
        if cuda_device == -1:
            return torch.from_numpy(data_structure)
        else:
            return torch.from_numpy(data_structure).cuda(cuda_device)
