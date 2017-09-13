#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling kernel"
/usr/local/cuda/bin/nvcc -c -o highway_lstm_kernel.cu.o highway_lstm_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../
python build.py
