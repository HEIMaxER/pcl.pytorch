#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda-9.0/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61

cd ../
python build.py
