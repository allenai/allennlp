# pylint: disable=invalid-name
import os
import torch
from torch.utils.ffi import create_extension

if not torch.cuda.is_available():
    raise Exception('HighwayLSTM can only be compiled with CUDA')

sources = ['src/highway_lstm_cuda.c']
headers = ['src/highway_lstm_cuda.h']
defines = [('WITH_CUDA', None)]
with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
extra_objects = ['src/highway_lstm_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
        '_ext.highway_lstm_layer',
        headers=headers,
        sources=sources,
        define_macros=defines,
        relative_to=__file__,
        with_cuda=with_cuda,
        extra_objects=extra_objects
        )

if __name__ == '__main__':
    ffi.build()
