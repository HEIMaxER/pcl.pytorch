#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda-9.0/
cd src
echo "Compiling stnm kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu -I${CUDA_PATH}/include -x cu -Xcompiler -fPIC -arch=sm_52
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!nms make!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
cd ../
python build.py
