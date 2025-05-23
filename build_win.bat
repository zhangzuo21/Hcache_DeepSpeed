@echo off

set CUDA_HOME=%CUDA_PATH%
set DISTUTILS_USE_SDK=1

set DS_BUILD_AIO=0
set DS_BUILD_CUTLASS_OPS=0
set DS_BUILD_EVOFORMER_ATTN=0
set DS_BUILD_FP_QUANTIZER=0
set DS_BUILD_GDS=0
set DS_BUILD_RAGGED_DEVICE_OPS=0
set DS_BUILD_SPARSE_ATTN=0
set DS_BUILD_DEEP_COMPILE=0

python -m build --wheel --no-isolation

:end
